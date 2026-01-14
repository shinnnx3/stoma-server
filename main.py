import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from supabase import create_client
import os
from datetime import datetime, timedelta
import shutil # 파일 저장용
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# ==========================================
# CORS 설정 (Vercel + ngrok 연동)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://stoma-care-buddy.vercel.app",
        "https://*.vercel.app",
        "*"  # 모든 origin 허용 (개발용)
    ],
    allow_credentials=False,  # ngrok + Vercel에서는 False로 설정
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ==========================================
# [1] 설정 및 경로 지정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "models", "best.pt")           
EFF_PATH = os.path.join(BASE_DIR, "models", "efficientnet.pth")

# Supabase 설정 (보내주신 키 적용됨)
SUPABASE_URL = "https://uvlfxtacgpkixdnbdibu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV2bGZ4dGFjZ3BraXhkbmJkaWJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgxMDg4MzYsImV4cCI6MjA4MzY4NDgzNn0.tkcuF4lT3pHyU27ewCDoaR5aHLIW3EBw-5zXCo1PakM"

BUCKET_NAME = "wound_images"
TABLE_NAME = "diagnosis_logs"

# 로컬 저장소 설정 (테스트용)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# [2] 이미지 처리 클래스
# ==========================================
class MedicalImageProcessor:
    def __init__(self, yolo_path, efficientnet_path, device="cpu"):
        self.device = torch.device(device)
        self.target_names = {"STOMA": "Stoma", "REF": "Tissue"}

        # 1. YOLO 로드
        print(f"🔄 Loading YOLO from {yolo_path}...")
        try:
            self.yolo_model = YOLO(yolo_path)
            print("✅ YOLO 로드 성공")
        except Exception as e:
            print(f"❌ YOLO Error: {e}")
            raise RuntimeError("YOLO 모델 로드 실패")
        
        # 2. EfficientNet 로드
        print(f"🔄 Loading EfficientNet from {efficientnet_path}...")
        try:
            self.classifier = models.efficientnet_b0(weights=None)
            num_ftrs = self.classifier.classifier[1].in_features
            self.classifier.classifier[1] = nn.Linear(num_ftrs, 4) 
            
            checkpoint = torch.load(efficientnet_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.classifier.load_state_dict(new_state_dict, strict=False)
            self.classifier.to(self.device)
            self.classifier.eval()
            print("✅ EfficientNet 로드 성공!")

        except Exception as e:
            print(f"❌ EfficientNet 로드 실패: {e}")
            self.classifier = None 

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_map = {0: 1, 1: 2, 2: 3, 3: 4}

    def _bytes_to_cv2(self, file_bytes):
        nparr = np.frombuffer(file_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _cv2_to_bytes(self, img):
        success, encoded_img = cv2.imencode('.jpg', img)
        return encoded_img.tobytes() if success else None

    def _is_valid_box(self, box, img_shape):
        if box is None: return False
        h, w = img_shape[:2]
        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return False
        if x2 <= x1 or y2 <= y1: return False
        return True

    def _calculate_wb_scale(self, img, ref_box):
        if not self._is_valid_box(ref_box, img.shape): return 1.0, 1.0, 1.0
        x1, y1, x2, y2 = map(int, ref_box)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return 1.0, 1.0, 1.0

        target = 240.0
        sb = min(target / (np.mean(roi[:,:,0]) + 1e-5), 3.0)
        sg = min(target / (np.mean(roi[:,:,1]) + 1e-5), 3.0)
        sr = min(target / (np.mean(roi[:,:,2]) + 1e-5), 3.0)
        return sb, sg, sr

    def _apply_wb_crop_clahe(self, img, stoma_box, scales):
        sb, sg, sr = scales
        x1, y1, x2, y2 = map(int, stoma_box)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None

        b, g, r = cv2.split(crop)
        b = cv2.multiply(b, sb); g = cv2.multiply(g, sg); r = cv2.multiply(r, sr)
        b, g, r = [np.clip(c, 0, 255).astype(np.uint8) for c in [b, g, r]]
        
        lab = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b_ch)), cv2.COLOR_LAB2BGR)

    def process(self, file_bytes):
        original_img = self._bytes_to_cv2(file_bytes)
        if original_img is None: return {"is_valid": False}

        results = self.yolo_model.predict(original_img, verbose=False, conf=0.2)
        result = results[0]

        def get_box(lbl):
            best, max_conf = None, 0.0
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if result.names[cls_id] == lbl and box.conf[0] > max_conf:
                    max_conf = float(box.conf[0]); best = box.xyxy[0].tolist()
            return best

        stoma_box = get_box(self.target_names["STOMA"])
        ref_box = get_box(self.target_names["REF"])

        if not self._is_valid_box(stoma_box, original_img.shape):
            hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            return {
                "necrosis_class": 0,
                "brightness": float(np.mean(hsv[:,:,2])),
                "processed_bytes": file_bytes,
                "is_valid": True,
                "note": "No stoma"
            }

        scales = self._calculate_wb_scale(original_img, ref_box)
        processed_img = self._apply_wb_crop_clahe(original_img, stoma_box, scales)
        if processed_img is None: return {"is_valid": False}

        necrosis_class = 1
        if self.classifier:
            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                _, predicted = torch.max(outputs, 1)
                necrosis_class = self.class_map.get(int(predicted.item()), 1)

        brightness = float(np.mean(cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)[:,:,2]))

        processed_bytes = self._cv2_to_bytes(processed_img)
        return {
            "necrosis_class": necrosis_class,
            "brightness": brightness,
            "processed_bytes": processed_bytes if processed_bytes else file_bytes,
            "is_valid": True
        }

# ==========================================
# [3] 서버 API (앱 초기화는 위에서 했음)
# ==========================================
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
processor = MedicalImageProcessor(YOLO_PATH, EFF_PATH)

@app.get("/")
def read_root():
    return JSONResponse(
        content={"message": "Stoma Care Server Running"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        # 0. 파일 로컬 저장 (디버깅용)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.seek(0)

        # 1. 이미지 처리
        contents = await file.read()
        result = processor.process(contents)

        if not result["is_valid"]:
            original_bytes = contents
            processed_bytes = contents
            current_brightness = 0.0
            necrosis_class = 0
        else:
            original_bytes = contents
            processed_bytes = result.get("processed_bytes") or contents  # None이면 원본 사용
            current_brightness = result["brightness"]
            necrosis_class = result["necrosis_class"]

        # 2. Supabase 스토리지에 이미지 업로드 (원본 + 처리본)
        timestamp = int(datetime.now().timestamp())

        # 2-1. 원본 이미지 업로드
        original_filename = f"original_{user_id}_{timestamp}.jpg"
        supabase.storage.from_(BUCKET_NAME).upload(
            path=original_filename,
            file=original_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        original_url = supabase.storage.from_(BUCKET_NAME).get_public_url(original_filename)

        # 2-2. 처리된 이미지 업로드
        corrected_filename = f"corrected_{user_id}_{timestamp}.jpg"
        supabase.storage.from_(BUCKET_NAME).upload(
            path=corrected_filename,
            file=processed_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        corrected_url = supabase.storage.from_(BUCKET_NAME).get_public_url(corrected_filename)

        # 3. 응답 데이터 (DB 저장은 프론트엔드에서 문진 완료 후 수행)
        response_data = {
            "status": "success",
            "data": {
                "original_image_url": original_url,
                "corrected_image_url": corrected_url,
                "necrosis_class": int(necrosis_class),
                "brightness": round(current_brightness, 1)
            },
            "message": "이미지 분석 완료"
        }

        # CORS 헤더를 명시적으로 추가
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        print(f"Server Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
