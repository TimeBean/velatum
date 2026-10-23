import os
import sys

from anonymizer import FaceAnonymizer, AnonymizeMethod


def main() -> None:
    input_file = 'input.png'
    output_file = 'output.png'
    model_file = 'face_detection_yunet_2023mar.onnx'

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found in the current directory.")
        sys.exit(1)

    try:
        anonymizer = FaceAnonymizer(model_path=model_file)

        # Выбор метода передается через параметр method
        anonymizer.process_image(
            input_path=input_file,
            output_path=output_file,
            method=AnonymizeMethod.BLUR,
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
