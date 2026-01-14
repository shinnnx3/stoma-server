# Railway 배포 가이드

## 🚀 빠른 시작

### 1. Railway CLI 설치
```bash
npm install -g @railway/cli
```

### 2. Railway 로그인
```bash
railway login
```

### 3. 프로젝트 초기화 및 배포
```bash
railway init
railway up
```

## 📋 배포 전 체크리스트

### ✅ 필수 파일 확인
- [x] `requirements.txt` - Python 패키지 목록
- [x] `Procfile` - 서버 실행 명령
- [x] `railway.toml` - Railway 설정
- [x] `main.py` - FastAPI 앱
- [x] `models/best.pt` - YOLO 모델
- [x] `models/efficientnet.pth` - EfficientNet 모델

### ⚠️ 주의사항

#### 1. 모델 파일 크기
- YOLO와 EfficientNet 모델 파일이 큽니다
- Railway는 Git으로 배포하므로 모델 파일이 Git에 포함되어야 합니다
- Git LFS 사용을 권장합니다:
  ```bash
  git lfs install
  git lfs track "*.pt"
  git lfs track "*.pth"
  git add .gitattributes
  git add models/
  git commit -m "Add model files with Git LFS"
  ```

#### 2. 환경 변수 설정
Railway 대시보드에서 다음 환경 변수를 설정하세요:
```bash
SUPABASE_URL=https://uvlfxtacgpkixdnbdibu.supabase.co
SUPABASE_KEY=your_supabase_key_here
```

#### 3. 메모리 요구사항
- PyTorch + YOLO + EfficientNet은 최소 2GB RAM 필요
- Railway 무료 티어는 512MB-1GB RAM 제한
- **Hobby Plan ($5/월) 이상 권장**

#### 4. 빌드 시간
- 첫 배포 시 10-15분 소요 (PyTorch, torchvision 설치)
- 이후 배포는 캐시로 빨라집니다

## 🔧 배포 후 설정

### 1. 도메인 확인
```bash
railway domain
```

### 2. 로그 확인
```bash
railway logs
```

### 3. 환경 변수 설정
```bash
railway variables set SUPABASE_KEY=your_key_here
```

## 🐛 트러블슈팅

### 빌드 실패 시
```bash
# 로그 확인
railway logs

# 흔한 문제:
# 1. 메모리 부족 → Hobby Plan으로 업그레이드
# 2. 모델 파일 누락 → Git LFS 확인
# 3. Python 버전 불일치 → railway.toml 확인
```

### 서버 시작 실패 시
```bash
# 헬스 체크
curl https://your-app.railway.app/

# API 문서 확인
curl https://your-app.railway.app/docs
```

## 💰 예상 비용

### Hobby Plan ($5/월)
- 8GB RAM
- 8 vCPU
- 무제한 대역폭
- **이 프로젝트에 권장**

### Pro Plan ($20/월)
- 32GB RAM
- 32 vCPU
- 우선 지원

## 🔗 유용한 링크

- Railway 대시보드: https://railway.app/dashboard
- Railway 문서: https://docs.railway.app/
- FastAPI 문서: https://fastapi.tiangolo.com/

## 📝 Git 커밋 가이드

```bash
# 모든 파일 추가
git add .

# 커밋
git commit -m "Add Railway deployment configuration"

# Railway 자동 배포 (main 브랜치에 push)
git push origin main
```

## 🎯 다음 단계

1. Railway 대시보드에서 환경 변수 설정
2. 커스텀 도메인 연결 (선택사항)
3. 모니터링 설정 (로그, 메트릭)
4. CI/CD 파이프라인 구성 (GitHub Actions)
