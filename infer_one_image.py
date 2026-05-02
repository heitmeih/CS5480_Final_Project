import sys

import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

from main import create_backbone

MODEL_PATH = "mannequin-tent-cnn-cs5480.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPINGS: dict[int, str] = {
    1: "person",
    2: "tent",
}


def inference(file_name: str) -> None:
    # Make the backbone
    model = create_backbone()
    model.to(DEVICE)

    # Load model weights after done training
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )

    model.eval()

    image = Image.open(file_name).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_tensor = to_tensor(image).to(DEVICE)

    with torch.no_grad():
        all_predictions = model([image_tensor])[0]

    mask = all_predictions["scores"] > 0.5
    boxes = all_predictions["boxes"][mask]
    labels = [
        CLASS_MAPPINGS.get(label.item(), "unknown")
        for label in all_predictions["labels"][mask]
    ]
    scores = all_predictions["scores"][mask]
    predictions = list(zip(boxes, labels, scores))
    for box, label, score in predictions:
        draw.rectangle(box.tolist(), outline="yellow", width=1)
        text_pt = (box.tolist()[0], box.tolist()[1] - 12)
        draw.text(text_pt, f"{label} ({score:.2f})", fill="red")

    image.show()


if __name__ == "__main__":
    file_name: str = sys.argv[1]
    inference(file_name)
