import os
from enum import Enum
import cv2
import numpy as np

from src.strategies import AnonymizeStrategy, BlurStrategy, FillStrategy


class AnonymizeMethod(Enum):
    BLUR = "blur"
    FILL_BLACK = "fill_black"
    FILL_WHITE = "fill_white"


def _process_face(image: np.ndarray, face: np.ndarray, strategy: AnonymizeStrategy) -> None:
    h, w = image.shape[:2]

    box = face[0:4].astype(int)
    x, y, box_w, box_h = box

    x = max(0, x)
    y = max(0, y)
    box_w = min(w - x, box_w)
    box_h = min(h - y, box_h)

    if box_w <= 0 or box_h <= 0:
        return

    strategy.apply(image, x, y, box_w, box_h)


class FaceAnonymizer:
    def __init__(self, model_path: str, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._strategies: dict[AnonymizeMethod, AnonymizeStrategy] = {
            AnonymizeMethod.BLUR: BlurStrategy(),
            AnonymizeMethod.FILL_BLACK: FillStrategy(color=(0, 0, 0)),
            AnonymizeMethod.FILL_WHITE: FillStrategy(color=(1, 1, 1))
        }

    def _init_detector(self, width: int, height: int):
        return cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(width, height),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    def process_image(self, input_path: str, output_path: str, method: AnonymizeMethod = AnonymizeMethod.BLUR) -> bool:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Failed to load image '{input_path}'.")
            return False

        h, w, _ = image.shape
        detector = self._init_detector(w, h)

        status, faces = detector.detect(image)

        if faces is None:
            print("Faces found: 0")
            faces = []
        else:
            print(f"Faces found: {len(faces)}")

        strategy = self._strategies.get(method)
        if not strategy:
            raise ValueError(f"Unknown anonymize method: {method}")

        for face in faces:
            _process_face(image, face, strategy)

        cv2.imwrite(output_path, image)
        print(f"Result saved to '{output_path}'.")
        return True