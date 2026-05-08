import sys
import time

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from train_resnet import CLASS_MAPPINGS, DEVICE, MODEL_PATH, create_backbone


def inference(file_name: str) -> None:
    # Get backbone from training file
    model = create_backbone()
    model.to(DEVICE)

    # Load model weights from training
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )

    model.eval()

    image = Image.open(file_name).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_tensor = to_tensor(image).to(DEVICE)

    print("Predicting...")
    # time the inference to get an idea of how long it takes to run
    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_predictions = model([image_tensor])[0]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    print(f"Inference time: {(t1 - t0) * 1000:.1f} ms")
    print(all_predictions)
    mask = all_predictions["scores"] > 0.5
    boxes = all_predictions["boxes"][mask]
    labels = [
        CLASS_MAPPINGS.get(label.item(), "unknown")
        for label in all_predictions["labels"][mask]
    ]
    scores = all_predictions["scores"][mask]
    predictions = list(zip(boxes, labels, scores))
    for box, label, score in predictions:
        draw.rectangle(box.tolist(), outline="yellow", width=8)
        text_pt = (box.tolist()[0], box.tolist()[1] - 60)
        draw.text(text_pt, f"{label} ({score:.2f})", fill="yellow", font_size=50)

    image.show()
    image.save("output.jpg")


if __name__ == "__main__":
    file_name: str = sys.argv[1]
    inference(file_name)
