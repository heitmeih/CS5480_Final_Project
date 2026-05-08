from pathlib import Path

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

from data import all_data_sources, data_dir, download_sources, load_datasets

MODEL_PATH = "mannequin-tent-cnn-cs5480-3.pth"

# parameters to edit
# faster r-cnn has a background class included by default, so its mannequin + tent + 1
NUM_CLASSES = 3  # 0 = background, 1 = person, 2 = tent
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPINGS: dict[int, str] = {
    1: "person",
    2: "tent",
}


class FasterRCNNDataset(Dataset):
    """
    Converts the ConcatDataset that can be retrieved from data.load_datasets()
    into a dataset that returns annotations into the format that is expected by
    the Faster R-CNN model.
    """

    def __init__(self, subfolder: str) -> None:
        self.dataset = load_datasets(subfolders=[subfolder])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, annotations = self.dataset[index]
        targets = self._convert_annotations(annotations)
        return image, targets

    def _convert_annotations(self, annotations: list[dict]) -> dict:
        # While the original dataset used xywh, we need to convert it to
        # xyxy format for Faster R-CNN. We also shift category ids by +1
        # to reserve the 0 label for the included xbackground class.
        boxes: list[list[float]] = []
        labels: list[int] = []
        iscrowd: list[int] = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

            # if the new box has negative area, skip it
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])

            # +1 shifts COCO ids (0-indexed) so that 0 is the background class
            labels.append(int(ann["category_id"]) + 1)
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            return {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }
        else:
            # no annotations to return, so put all 0's to show the model that
            # there is nothing to predict
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }


def collate_fn(batch):
    """Keep images and targets as tuples instead of stacking them"""
    return tuple(zip(*batch))


def create_backbone() -> FasterRCNN:
    """
    Creates a base FasterRCNN model with pre-trained weights that can
    then be fine-tuned on our dataset.
    """
    # load faster-rcnn backbone using resnet50-fpn, with the default weights
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )

    # Replace the default classification layer with our own to support the custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    return model


def train_one_epoch(
    model,
    optimizer: AdamW,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Full training loop for one epoch.
    """
    model.train()  # set model to training mode

    total_loss = 0.0
    num_batches = len(data_loader)

    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # add loss of this batch to total loss
        batch_loss = losses.item()
        total_loss += batch_loss

        # status print
        print(f"  [step {step:4d}/{num_batches}]  loss: {batch_loss:.4f}", end="\r")

    # return the total loss for the epoch, averaged over all batches
    return total_loss / max(num_batches, 1)


def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Run the validation loop over the given validation set
    """
    model.train()

    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():  # Disable gradient computation since this is validation
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses

    return total_loss / max(num_batches, 1)


def main() -> None:
    # Right now this will download the data, train the model, and save the weights

    if not Path(data_dir).exists():
        download_sources(all_data_sources, data_dir)

    # load the train/val dataset into FasterRCNNDataset
    print("Loading datasets …")
    train_dataset = FasterRCNNDataset("train")
    print(f"Training samples: {len(train_dataset)}")
    val_dataset = FasterRCNNDataset("valid")
    print(f"Validation samples: {len(val_dataset)}")

    # load training parameters
    use_pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    # Make the backbone
    model = create_backbone()
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}  (lr={scheduler.get_last_lr()})")

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch}  |  train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}"
        )

    # Save model weights after done training
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
