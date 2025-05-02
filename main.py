from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# Cấu hình CORS để cho phép truy cập từ FE (ví dụ: localhost:5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Hoặc dùng ["*"] để test thoải mái
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model từ file
with open("model.pkl", "rb") as f:
    model, _ = pickle.load(f)

# Route xử lý POST để dự đoán
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        print("Dữ liệu nhận được:", data)  # In ra dữ liệu từ FE để xem chính xác nó như thế nào

        # Kiểm tra nếu có trường "landmarks"
        landmarks = data.get("landmarks")
        if not landmarks:
            return JSONResponse(content={"error": "Thiếu landmarks"}, status_code=422)

        # Kiểm tra xem số lượng landmarks có đủ 42 không
        if len(landmarks) != 42:
            return JSONResponse(content={"error": "Cần đủ 42 landmarks"}, status_code=422)

        # Kiểm tra từng landmark có đủ 3 thuộc tính x, y, z không
        for i, lm in enumerate(landmarks):
            if not all(k in lm for k in ["x", "y", "z"]):
                return JSONResponse(content={"error": f"Landmark {i+1} thiếu x, y, hoặc z"}, status_code=422)

        # Chuyển các landmark thành mảng phẳng [x1, y1, z1, x2, y2, z2,...]
        flat = [coord for lm in landmarks for coord in (lm["x"], lm["y"], lm["z"])]

        # Kiểm tra kích thước mảng sau khi phẳng
        if len(flat) != 42 * 3:  # Đảm bảo có đúng 42 landmarks x 3 (x, y, z)
            return JSONResponse(content={"error": "Kích thước dữ liệu không đúng"}, status_code=422)

        # Dự đoán với model
        prediction = model.predict([flat])[0]
        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Route xử lý OPTIONS để fix lỗi 405 Method Not Allowed
@app.options("/predict")
async def preflight():
    return Response(status_code=204)
