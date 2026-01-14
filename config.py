# config.py
from supabase import create_client, Client

# ============================
# 1. 내 정보 입력 (여기를 꼭 채우세요!)
# ============================
SUPABASE_URL = "https://uvlfxtacgpkixdnbdibu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV2bGZ4dGFjZ3BraXhkbmJkaWJ1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjgxMDg4MzYsImV4cCI6MjA4MzY4NDgzNn0.tkcuF4lT3pHyU27ewCDoaR5aHLIW3EBw-5zXCo1PakM"


# 2. 사용할 이름들
BUCKET_NAME = "wound_images"       # Storage 이름
TABLE_NAME = "diagnosis_logs"      # Table 이름

# 3. Supabase 연결 (여기서 연결을 미리 다 해둡니다)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)