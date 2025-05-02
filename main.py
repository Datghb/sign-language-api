from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# Cấu hình CORS (tạm thời allow tất cả để test thoải mái)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model từ file
try:
    with open("model.pkl", "rb") as f:
        model, _ = pickle.load(f)
    print("✅ Model đã được load thành công.")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    model = None

# Route dự đoán
@app.post("/predict")
async def predict(request: Request):
    if model is None:
        return JSONResponse(content={"error": "Model chưa được load"}, status_code=500)
    try:
        data = await request.json()
        print("📥 Dữ liệu nhận được:", data)

        landmarks = data.get("landmarks")
        if not landmarks:
            return JSONResponse(content={"error": "Thiếu landmarks"}, status_code=422)
        if len(landmarks) != 42:
            return JSONResponse(content={"error": "Cần đủ 42 landmarks"}, status_code=422)

        for i, lm in enumerate(landmarks):
            if not all(k in lm for k in ["x", "y", "z"]):
                return JSONResponse(content={"error": f"Landmark {i+1} thiếu x, y, hoặc z"}, status_code=422)

        flat = [coord for lm in landmarks for coord in (lm["x"], lm["y"], lm["z"])]
        if len(flat) != 42 * 3:
            return JSONResponse(content={"error": "Kích thước dữ liệu không đúng"}, status_code=422)

        prediction = model.predict([flat])[0]
        print("🎯 Dự đoán:", prediction)

        # Ép kiểu về int chuẩn trước khi trả về JSON
        return JSONResponse(content={"prediction": int(prediction)})


    except Exception as e:
        print("❌ Lỗi xử lý:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Xử lý preflight CORS request
@app.options("/predict")
async def preflight():
    return Response(status_code=204)
