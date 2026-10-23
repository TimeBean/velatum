import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from anonymizer import FaceAnonymizer, AnonymizeMethod

app = FastAPI(title="Face Blur API", description="API for blurring faces in photos")

MODEL_FILE = 'face_detection_yunet_2023mar.onnx'
try:
    anonymizer = FaceAnonymizer(model_path=MODEL_FILE)
except Exception as e:
    anonymizer = None
    print(f"Error loading model: {e}")

def cleanup_file(file_path: str) -> None:
    """Deletes the temporary file after sending the response to the client."""
    if os.path.exists(file_path):
        os.remove(file_path)

@app.post("/api/v1/blur-face")
async def blur_face(image: UploadFile = File(...)):
    if anonymizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    fd_in, temp_in_path = tempfile.mkstemp(suffix=".png")
    fd_out, temp_out_path = tempfile.mkstemp(suffix=".png")

    os.close(fd_in)
    os.close(fd_out)

    try:
        with open(temp_in_path, "wb") as f_in:
            content = await image.read()
            f_in.write(content)

        anonymizer.process_image(
            input_path=temp_in_path,
            output_path=temp_out_path,
            method=AnonymizeMethod.BLUR
        )

        cleanup_file(temp_in_path)

        return FileResponse(
            path=temp_out_path,
            media_type="image/png",
            filename=f"blurred_{image.filename}",
            background=BackgroundTask(cleanup_file, temp_out_path)
        )

    except Exception as e:
        cleanup_file(temp_in_path)
        cleanup_file(temp_out_path)
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")