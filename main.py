import os
import cv2
import sys


class FaceAnonymizer:
    """
    Detects and blurs faces in images using the YuNet model.
    """

    def __init__(self, model_path: str, score_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def _init_detector(self, width: int, height: int):
        return cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(width, height),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    def _anonimize_face(self, image, face):
        h, w = image.shape[:2]

        box = face[0:4].astype(int)
        x, y, box_w, box_h = box

        x = max(0, x)
        y = max(0, y)
        box_w = min(w - x, box_w)
        box_h = min(h - y, box_h)

        if box_w <= 0 or box_h <= 0:
            return

        face_roi = image[y:y + box_h, x:x + box_w]

        # Ensure kernel dimensions are odd numbers
        kw = max(3, box_w // 2)
        kw = kw if kw % 2 != 0 else kw + 1

        kh = max(3, box_h // 2)
        kh = kh if kh % 2 != 0 else kh + 1

        replacement = cv2.GaussianBlur(face_roi, (kw, kh), 30)
        image[y:y + box_h, x:x + box_w] = replacement

    def process_image(self, input_path: str, output_path: str):
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

        for face in faces:
            self._anonimize_face(image, face)

        cv2.imwrite(output_path, image)
        print(f"Result saved to '{output_path}'.")
        return True


if __name__ == "__main__":
    INPUT_FILE = 'input.png'
    OUTPUT_FILE = 'output.png'
    MODEL_FILE = 'face_detection_yunet_2023mar.onnx'

    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found in the current directory.")
        sys.exit(1)

    try:
        anonymizer = FaceAnonymizer(model_path=MODEL_FILE)
        anonymizer.process_image(INPUT_FILE, OUTPUT_FILE)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)