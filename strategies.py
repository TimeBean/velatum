from abc import ABC, abstractmethod
import cv2
import numpy as np


class AnonymizeStrategy(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        pass


class BlurStrategy(AnonymizeStrategy):
    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        face_roi = image[y:y + h, x:x + w]

        kw = max(3, w // 2)
        kw = kw if kw % 2 != 0 else kw + 1

        kh = max(3, h // 2)
        kh = kh if kh % 2 != 0 else kh + 1

        image[y:y + h, x:x + w] = cv2.GaussianBlur(face_roi, (kw, kh), 30)


class FillStrategy(AnonymizeStrategy):
    def __init__(self, color: tuple[int, int, int] = (0, 0, 0)):
        self.color = color

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        image[y:y + h, x:x + w] = self.color