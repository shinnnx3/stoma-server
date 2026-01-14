# image_processing.py

import cv2

import numpy as np

import torch

from ultralytics import YOLO

from torchvision import transforms

from PIL import Image

import os



class MedicalImageProcessor:

    def __init__(self):

        # 1. 모델 경로 설정 (현재 파일 위치 기준 models 폴더)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.yolo_path = os.path.join(BASE_DIR, "models", "best.pt")

        self.eff_path = os.path.join(BASE_DIR, "models", "efficientnet.pth")

        

        self.device = torch.device("cpu") # GPU 없으면 cpu 사용

        self.target_names = {"STOMA": "Stoma", "REF": "Tissue"}

        

        # 2. YOLO 로드

        print(f"🔄 Loading YOLO from {self.yolo_path}...")

        try:

            self.yolo_model = YOLO(self.yolo_path)

            print("✅ YOLO 로드 성공")

        except Exception as e:

            print(f"⚠️ YOLO 로드 실패: {e}")

            self.yolo_model = None



        # 3. EfficientNet 로드

        print(f"🔄 Loading EfficientNet from {self.eff_path}...")

        self.classifier = None

        try:

            self.classifier = torch.load(self.eff_path, map_location=self.device)

            self.classifier.eval()

            print("✅ EfficientNet 로드 성공")

        except Exception as e:

            print(f"⚠️ EfficientNet 로드 실패 (임시 모드로 동작): {e}")

        # 전처리 도구
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _bytes_to_cv2(self, file_bytes):
        nparr = np.frombuffer(file_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _cv2_to_bytes(self, img):
        success, encoded_img = cv2.imencode('.jpg', img)
        return encoded_img.tobytes() if success else None

    # (팀원이 준 화이트밸런스 계산 함수)
    def _calculate_wb_scale(self, img, ref_box):
        if ref_box is None: return 1.0, 1.0, 1.0
        x1, y1, x2, y2 = map(int, ref_box)
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return 1.0, 1.0, 1.0
        
        target = 240.0
        sb = min(target / (np.mean(roi[:,:,0]) + 1e-5), 3.0)
        sg = min(target / (np.mean(roi[:,:,1]) + 1e-5), 3.0)
        sr = min(target / (np.mean(roi[:,:,2]) + 1e-5), 3.0)
        return sb, sg, sr

    # (팀원이 준 보정 및 크롭 함수)
    def _apply_wb_crop_clahe(self, img, stoma_box, scales):
        sb, sg, sr = scales
        x1, y1, x2, y2 = map(int, stoma_box)
        h, w, _ = img.shape
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

    # ★ 메인 처리 함수
    def process(self, file_bytes):
        original_img = self._bytes_to_cv2(file_bytes)
        if original_img is None: return {"is_valid": False}

        # YOLO 모델이 없거나 로드 실패시 안전장치
        if self.yolo_model is None:
            return {
                "necrosis_class": 0,
                "brightness": 0.0,
                "processed_bytes": file_bytes,
                "is_valid": True,
                "note": "AI Model Not Loaded"
            }

        # 1. YOLO 예측
        results = self.yolo_model.predict(original_img, verbose=False, conf=0.2)
        result = results[0]

        # 박스 찾기 로직
        def get_box(lbl):
            best, max_conf = None, 0.0
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if result.names[cls_id] == lbl and box.conf[0] > max_conf:
                    best, max_conf = box.xyxy[0].tolist(), float(box.conf[0])
            return best

        stoma_box = get_box(self.target_names["STOMA"])
        ref_box = get_box(self.target_names["REF"])

        # Stoma 없으면 원본 리턴
        if stoma_box is None:
            hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            return {
                "necrosis_class": 0,
                "brightness": float(np.mean(hsv[:,:,2])),
                "processed_bytes": file_bytes,
                "is_valid": True,
                "note": "No stoma detected"
            }

        # 2. 보정 (WB -> Crop -> CLAHE)
        scales = self._calculate_wb_scale(original_img, ref_box)
        processed_img = self._apply_wb_crop_clahe(original_img, stoma_box, scales)
        
        if processed_img is None: return {"is_valid": False}

        # 3. EfficientNet 예측
        necrosis_class = 1
        if self.classifier:
            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                _, predicted = torch.max(outputs, 1)
                necrosis_class = int(predicted.item())

        # 4. 밝기 계산
        brightness = float(np.mean(cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)[:,:,2]))

        return {
            "necrosis_class": necrosis_class,
            "brightness": brightness,
            "processed_bytes": self._cv2_to_bytes(processed_img),
            "is_valid": True
        }
