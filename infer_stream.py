# Used for showcasing live inference using a webcam, for presentation day
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import torch
from torchvision.transforms.functional import to_tensor

from train_resnet import CLASS_MAPPINGS, DEVICE, MODEL_PATH, create_backbone


class Stream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame: Optional[cv2.typing.MatLike] = None
        self.thread = threading.Thread(target=self._reader)
        self.thread.start()

    def __del__(self):
        self.cap.release()

    def _reader(self):
        # Make sure the stream is open and continuously read frames
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            self.frame = frame

    def get_frame(self):
        while self.frame is None:
            pass
        return self.frame


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


def run_inference(
    model: torch.nn.Module, stream: Stream
) -> tuple[list[Detection], cv2.typing.MatLike]:
    frame = stream.get_frame()
    if frame is not None:
        image_tensor = to_tensor(frame).to(DEVICE)
        with torch.no_grad():
            all_predictions = model([image_tensor])[0]

        # filter detections
        mask = all_predictions["scores"] > 0.9
        boxes = all_predictions["boxes"][mask]
        labels = [
            CLASS_MAPPINGS.get(label.item(), "unknown")
            for label in all_predictions["labels"][mask]
        ]
        scores = all_predictions["scores"][mask]
        predictions: list[Detection] = [
            Detection(
                label=label,
                confidence=score,
                box=box.round().type(torch.int64).tolist(),
            )
            for box, label, score in list(zip(boxes, labels, scores))
        ]

        return predictions, frame
    else:
        raise RuntimeError


def infer_stream():
    # Create stream and backbone model
    stream = Stream()
    model = create_backbone()
    model.to(DEVICE)

    # Load model weights from training
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )

    model.eval()

    while True:
        predictions, frame = run_inference(model, stream)
        for pred in predictions:
            print(pred.label, pred.confidence, pred.box)
            print(pred.box[:2], pred.box[2:])
            print(type(frame))
            cv2.rectangle(frame, pred.box[:2], pred.box[2:], (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{pred.label} ({pred.confidence:.2f})",
                (pred.box[0], pred.box[1] - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    infer_stream()
