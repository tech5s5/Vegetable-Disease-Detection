from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

MODEL = tf.keras.models.load_model('saved_models/1')
CLASS_NAMES = ['Potato Early Blight','Potato Late Blight','Healthy Potato','Tomato Early Blight','Tomato Late Blight','Healthy Tomato']
IMG_SIZE = 256

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    """
    Reads uploaded image bytes and converts them into
    a (256, 256, 3) numpy array WITHOUT normalization
    """
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Resize ONLY
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy (uint8: 0–255)
    image = np.array(image)

    return image


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Add batch dimension → (1, 256, 256, 3)
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }