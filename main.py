# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()

# CORS cho phép FE gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def read_imagefile(file) -> np.ndarray:
    image = Image.open(BytesIO(file))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = read_imagefile(image_bytes)

    # Resize, preprocess ảnh theo cách mà cậu đã dùng khi train
    img_resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    prediction = model.predict(img_resized)[0]
    return {"result": prediction}
