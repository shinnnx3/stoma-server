from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import random
from datetime import datetime

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Form(...)):
    ts = int(datetime.now().timestamp())
    brightness = round(random.uniform(50.0, 220.0), 1)
    necrosis_class = random.randint(1, 4)
    response = {
        "status": "success",
        "data": {
            "original_image_url": f"https://example.com/original_{user_id}_{ts}.jpg",
            "corrected_image_url": f"https://example.com/corrected_{user_id}_{ts}.jpg",
            "necrosis_class": necrosis_class,
            "brightness": brightness,
        },
        "message": "mock upload ok",
    }
    return JSONResponse(content=response)

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Form(...)):
    # 임시 샘플 응답: 밝기 랜덤, 클래스 랜덤, 이미지 URL 더미
    ts = int(datetime.now().timestamp())
    brightness = round(random.uniform(50.0, 220.0), 1)
    necrosis_class = random.randint(1,4)
    response = {
        "status": "success",
        "data": {
            "original_image_url": f"https://example.com/original_{user_id}_{ts}.jpg",
            "corrected_image_url": f"https://example.com/corrected_{user_id}_{ts}.jpg",
            "necrosis_class": necrosis_class,
            "brightness": brightness
        },
        "message": "mock upload ok"
    }
    return JSONResponse(content=response)