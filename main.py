from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import pickle

from pygments.formatters import img

app = FastAPI()

# Load model.pkl
# Khi load model, giả sử bạn load tuple
with open("model.pkl", "rb") as f:
    model, _ = pickle.load(f)  # Trích xuất model từ tuple

# Sau khi lấy model, bạn có thể gọi predict
prediction = model.predict([img])  # Lúc này sẽ không bị lỗi nữa

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Xử lý ảnh với model
    prediction = model.predict([img])  # Giả sử model của bạn dự đoán một ảnh

    # Trả kết quả nhận diện
    return JSONResponse(content={"prediction": prediction})
