# triage.py
# 문진(Triage) 알고리즘 & 처방 데이터베이스 (순서 변경 적용)

def get_next_step(current_step_id, answer_index, ai_class, saved_diagnosis=""):
    """
    current_step_id: 현재 단계 ID
    answer_index: 사용자가 선택한 답변 번호
    ai_class: 사진 분석 결과 클래스 (1, 2, 3 등)
    saved_diagnosis: 이전 단계까지 누적된 진단명
    """

    # ----------------------------------------------------
    # ★ [변경] 0. 시작 (START) -> 즉시 사진 촬영 요청
    # ----------------------------------------------------
    if current_step_id == "START":
        return {
            "type": "camera_request",
            "message": "진단을 시작합니다. 먼저 장루 상태를 확인할 수 있도록 사진을 촬영해주세요.",
            "action": "open_camera",
            "next_step_id": "E_Q1" # 사진 찍은 후 응급 문진 1번으로 연결
        }

    # ----------------------------------------------------
    # 1. [Class 0] 응급 선별 (Emergency) - 사진 촬영 후 진행
    # ----------------------------------------------------
    if current_step_id == "E_Q1": 
        if answer_index == 0: return final_result("활동성 출혈")
        else: return questions["E_Q2"]

    if current_step_id == "E_Q2": 
        if answer_index == 0: return final_result("장폐색")
        else: return questions["E_Q3"]

    if current_step_id == "E_Q3": 
        if answer_index == 0: return final_result("급성 괴사")
        else: return questions["E_Q4"]
        
    if current_step_id == "E_Q4": 
        if answer_index == 0: return final_result("감돈된 장루 탈출")
        else: 
            # 응급 아님 -> AI 분석 결과(ai_class)에 따른 2차 문진으로 이동
            if ai_class == 1: return questions["C1_Q1"]
            elif ai_class == 2: return questions["C2_Q1"]
            elif ai_class == 3: return questions["C3_Q1"]
            else: return final_result("진단 불가 (AI 분석 오류)")

    # ----------------------------------------------------
    # ★ [사진 촬영 후] 2차 문진 시작점
    # ----------------------------------------------------
    if current_step_id == "POST_PHOTO_START":
        # AI 분석 결과(ai_class)에 따라 질문지 분기
        if ai_class == 1: return questions["C1_Q1"]
        elif ai_class == 2: return questions["C2_Q1"]
        elif ai_class == 3: return questions["C3_Q1"]
        else: return final_result("진단 불가 (AI 분석 오류)")

    # ----------------------------------------------------
    # 2. [Class 1] 정상/창백함
    # ----------------------------------------------------
    if current_step_id == "C1_Q1":
        if answer_index == 0: return to_common("허혈")
        elif answer_index == 1: return to_common("짓무름")
        else: return questions["C1_Q2"]

    if current_step_id == "C1_Q2":
        if answer_index == 0: return to_common("장루 협착")
        else: return to_common("정상")

    # ----------------------------------------------------
    # 3. [Class 2] 피부염/발적
    # ----------------------------------------------------
    if current_step_id == "C2_Q1":
        if answer_index == 0: return to_common("점막피부 분리")
        else: return questions["C2_Q2"]

    if current_step_id == "C2_Q2":
        if answer_index == 0: return questions["C2_Q2_SUB"]
        else: return questions["C2_Q3"]

    if current_step_id == "C2_Q2_SUB":
        if answer_index == 0: return to_common("단순 타박상")
        else: return to_common("외상성 궤양")

    if current_step_id == "C2_Q3":
        if answer_index == 0: return to_common("짓무름")
        elif answer_index == 1: return to_common("붉은 흉터")
        else: return questions["C2_Q4"]

    if current_step_id == "C2_Q4":
        if answer_index == 0: return to_common("요산 결정")
        elif answer_index == 1: return to_common("건선")
        elif answer_index == 2: return to_common("육아종")
        elif answer_index == 3: return to_common("곰팡이 감염")
        elif answer_index == 4: return to_common("알레르기")
        elif answer_index == 5: return to_common("기성 사마귀 병변")
        else: return questions["C2_Q5"]

    if current_step_id == "C2_Q5":
        cause = ""
        if answer_index == 0: cause = "자극성 피부염"
        elif answer_index == 1: cause = "기계적 손상"
        elif answer_index == 2: cause = "모낭염"
        else: 
            # 상세 불명 -> SACS 생략하고 바로 병원 권유
            return to_common("상세 불명의 발적")

        next_q = questions["C2_SACS"].copy()
        next_q["temp_diagnosis"] = cause 
        return next_q

    if current_step_id == "C2_SACS":
        sacs_score = ""
        if answer_index == 0: sacs_score = "[L1]"
        elif answer_index == 1: sacs_score = "[L2]"
        elif answer_index == 2: sacs_score = "[L3]"
        
        combined_diag = f"{saved_diagnosis} {sacs_score}"
        return to_common(combined_diag)

    # ----------------------------------------------------
    # 4. [Class 3] 혈류 및 변색 (★재촬영 로직 포함)
    # ----------------------------------------------------
    if current_step_id == "C3_Q1": 
        # 0번(그림자), 1번(굳은 피) -> 재촬영 및 사진 삭제 요청
        if answer_index == 0: return retry_result("단순 그림자입니다. 조명을 확인하고 다시 촬영해주세요.")
        elif answer_index == 1: return retry_result("굳은 피(출혈)입니다. 피를 닦아낸 후 다시 촬영해주세요.")
        
        elif answer_index == 2: return to_common("대장흑색증")
        elif answer_index == 3: return to_common("흉터/색소침착")
        else: return questions["C3_Q2"]

    if current_step_id == "C3_Q2": 
        if answer_index == 0: return questions["C3_Q2_SUB"]
        else: return questions["C3_Q3"]

    if current_step_id == "C3_Q2_SUB": 
        if answer_index == 0: return to_common("단순 타박상 (진한 멍)")
        elif answer_index == 1: return to_common("괴저성 농피증")
        else: return to_common("괴사")

    if current_step_id == "C3_Q3": 
        if answer_index == 0: return to_common("장루 정맥류")
        elif answer_index == 1: return to_common("괴저성 농피증")
        elif answer_index == 2: return to_common("괴사")
        else: return questions["C2_Q1"]

    # ----------------------------------------------------
    # 5. [공통] 구조적 이상 체크 (★다중 진단 병합)
    # ----------------------------------------------------
    if current_step_id == "COMMON_Q1":
        structure_diag = ""
        if answer_index == 0: structure_diag = "장루 탈출"
        elif answer_index == 1: structure_diag = "장루 함몰"
        elif answer_index == 2: structure_diag = "장루 탈장"
        
        if structure_diag:
            if saved_diagnosis == "정상":
                final_text = structure_diag
            else:
                final_text = f"{saved_diagnosis}, {structure_diag}"
        else:
            final_text = saved_diagnosis

        return final_result(final_text)

    return final_result("알 수 없는 오류")


# ==========================================
# ★ 헬퍼 함수 총정리 (요청하신 기능 3가지 반영)
# ==========================================

# 1. 다중 진단 및 위험도(Risk Level) 계산 함수
def final_result(diagnosis_text):
    found_descriptions = []
    found_advices = []
    found_alerts = []
    
    # 0: 정상(초록), 1: 주의(노랑), 2: 위험(빨강)
    max_risk_level = 0 

    # 진단명에 포함된 모든 키워드를 순회하며 정보 수집
    for keyword, info in TREATMENT_DATA.items():
        if keyword in diagnosis_text:
            found_descriptions.append(info["description"])
            found_advices.append(info["advice"])
            
            # 위험도 로직
            if info["emergency_alert"]: 
                # 응급 알림이 있으면 무조건 '위험(2)'
                found_alerts.append(info["emergency_alert"])
                max_risk_level = 2
            elif max_risk_level < 2:
                # 응급은 아니지만, '정상'이나 '흉터' 같은 안전한 게 아니면 '주의(1)'
                safe_keywords = ["정상", "흉터", "대장흑색증", "단순 타박상"]
                is_safe = any(safe in keyword for safe in safe_keywords)
                
                if not is_safe:
                    max_risk_level = max(max_risk_level, 1)

    # 내용이 없을 때 예외처리
    if not found_descriptions:
        final_desc = "상세 진단을 위해 의료진과 상담이 필요합니다."
        final_advice = "가까운 병원을 방문하거나 전담 간호사에게 문의하세요."
        final_alert = None
        max_risk_level = 1 
    else:
        final_desc = "\n".join(found_descriptions)
        final_advice = "\n\n[추가 처방]\n".join(found_advices)
        final_alert = "\n".join(found_alerts) if found_alerts else None

    return {
        "type": "result",
        "diagnosis": diagnosis_text,
        "description": final_desc,
        "advice": final_advice,
        "emergency_alert": final_alert,
        "risk_level": max_risk_level # 0, 1, 2 중 하나 반환
    }

# 2. 재촬영 및 이미지 삭제 함수
def retry_result(message):
    return {
        "type": "retry",
        "diagnosis": message,
        "action": "camera",
        "delete_image": True # ★ 앱에서 이미지 삭제하도록 신호 보냄
    }

# 3. 공통 질문으로 넘기기
def to_common(diagnosis_so_far):
    q = questions["COMMON_Q1"].copy()
    q["temp_diagnosis"] = diagnosis_so_far
    return q

# ==========================================
# 질문 데이터베이스
# ==========================================
questions = {
    "E_Q1": { "id": "E_Q1", "type": "question", "text": "현재 피가 멈추지 않고 계속 흐르나요?", "options": ["예 (응급)", "아니요"] },
    "E_Q2": { "id": "E_Q2", "type": "question", "text": "24시간 이상 가스/대변 배출이 없고 복통/구토가 있나요?", "options": ["예", "아니요"] },
    "E_Q3": { "id": "E_Q3", "type": "question", "text": "장루 색깔이 검게/보라색으로 변했거나 창백해졌나요?", "options": ["예", "아니요"] },
    "E_Q4": { "id": "E_Q4", "type": "question", "text": "장루가 길게 튀어나와 아프거나 피부 사이가 벌어졌나요?", "options": ["예", "아니요"] },

    "C1_Q1": { "id": "C1_Q1", "type": "question", "text": "장루와 주변 피부 색깔이 어떤가요?", "options": ["장루가 하얗게 변함", "피부만 하얗게 붊", "선홍색 정상"] },
    "C1_Q2": { "id": "C1_Q2", "type": "question", "text": "대변이 가늘게 나오거나 통증이 있나요?", "options": ["예", "아니요"] },

    "C2_Q1": { "id": "C2_Q1", "type": "question", "text": "장루와 피부 사이가 벌어져 속살이 보이나요?", "options": ["예", "아니요"] },
    "C2_Q2": { "id": "C2_Q2", "type": "question", "text": "최근 장루를 부딪히거나 눌린 적 있나요?", "options": ["예", "아니요"] },
    "C2_Q2_SUB": { "id": "C2_Q2_SUB", "type": "question", "text": "통증은 어떤가요?", "options": ["욱신거림/매끈함", "스치면 극심한 통증"] },
    "C2_Q3": { "id": "C2_Q3", "type": "question", "text": "붉은 부위 질감/시기가 어떤가요?", "options": ["쭈글쭈글/축축함", "오래된 자국", "최근/따가움"] },
    "C2_Q4": { "id": "C2_Q4", "type": "question", "text": "붉은 발진 외 특징이 있나요?", "options": ["하얀 모래알", "은백색 비늘", "붉은 살점", "좁쌀/가려움", "장루판 모양", "두꺼운 피부(사마귀)", "없음"] },
    
    "C2_Q5": { 
        "id": "C2_Q5", "type": "question", 
        "text": "평소 관리 상태가 어떤가요?", 
        "options": ["대변이 자주 샘(자극성)", "뗄 때 아픔(기계적)", "털구멍 뾰루지(모낭염)", "잘 모르겠음"] 
    },
    
    "C2_SACS": {
        "id": "C2_SACS", "type": "question",
        "text": "현재 피부 상태의 심각도를 골라주세요.",
        "options": ["빨갛기만 함 (L1)", "껍질이 벗겨짐 (L2)", "깊게 파이거나 보라색 (L3)"]
    },

    "C3_Q1": { "id": "C3_Q1", "type": "question", "text": "변색 부위 특징을 골라주세요.", "options": ["그림자", "닦으면 피 나옴", "변비약 복용", "오래된 흉터", "없음(이상함)"] },
    "C3_Q2": { "id": "C3_Q2", "type": "question", "text": "부딪히거나 다친 적 있나요?", "options": ["예", "아니요"] },
    "C3_Q2_SUB": { "id": "C3_Q2_SUB", "type": "question", "text": "통증과 감각은 어떤가요?", "options": ["욱신거리는 멍", "스치면 극심함", "차가움/감각없음"] },
    "C3_Q3": { "id": "C3_Q3", "type": "question", "text": "모양과 통증을 설명해주세요.", "options": ["지렁이 혈관", "보라색 테두리+통증", "통증 없음/검게 마름", "해당 없음"] },

    "COMMON_Q1": {
        "id": "COMMON_Q1", "type": "question",
        "text": "[마지막] 장루의 모양이나 높이가 평소와 다른가요?",
        "options": ["길게 튀어나옴(탈출)", "안으로 파고듦(함몰)", "기침 시 솟음(탈장)", "평소와 같음"]
    }
}

# ==========================================
# 처방전 데이터베이스
# ==========================================
TREATMENT_DATA = {
    "활동성 출혈": { "description": "지혈되지 않는 활동성 출혈이 감지되었습니다.", "advice": "출혈량이 많습니다. 깨끗한 거즈나 수건으로 부위를 압박하십시오.", "emergency_alert": "즉시 응급실로 가십시오!" },
    "장폐색": { "description": "장이 막힌 '장폐색'이 의심됩니다.", "advice": "따뜻한 물을 마시고, 가벼운 걷기나 시계 방향 마사지를 하세요. 음식 섭취를 중단하십시오.", "emergency_alert": "복통이 심해지거나 구토가 동반되면 즉시 응급실로 이동하십시오." },
    "급성 괴사": { "description": "조직이 죽어가는 '장루 괴사'가 의심됩니다.", "advice": "자가 관리로 회복될 수 없습니다. 괴사 부위를 제거하는 수술적 처치가 필요할 수 있습니다.", "emergency_alert": "장루 전체가 썩어 들어갈 위험이 있습니다. 지금 즉시 응급실로 가십시오!" },
    "괴사": { "description": "조직이 죽어가는 '장루 괴사'가 의심됩니다.", "advice": "자가 관리로 회복될 수 없습니다. 괴사 부위를 제거하는 수술적 처치가 필요할 수 있습니다.", "emergency_alert": "장루 전체가 썩어 들어갈 위험이 있습니다. 지금 즉시 응급실로 가십시오!" },
    "감돈된 장루 탈출": { "description": "장루가 심각하게 탈출되어 혈류가 차단될 위험이 있습니다.", "advice": "억지로 밀어 넣지 마시고 젖은 거즈로 덮으십시오.", "emergency_alert": "장루 색이 변하고 있습니다. 즉시 응급실로 가십시오!" },
    "괴저성 농피증": { "description": "자가면역 질환인 '괴저성 농피증'이 강력히 의심됩니다.", "advice": "절대 상처를 뜯거나 소독약으로 세게 닦지 마십시오. 일반 치료로 낫지 않으므로 병원 방문이 필요합니다.", "emergency_alert": "통증이 심하고 위험한 질환입니다. 즉시 병원으로 가십시오." },
    "허혈": { "description": "혈액 공급 부족으로 인한 '허혈'이 의심됩니다.", "advice": "자가 치료가 불가능한 상태입니다. 수술 부위 혈관 문제일 수 있습니다.", "emergency_alert": "장루 괴사로 진행될 수 있으니 지금 즉시 병원(응급실)에 연락하세요." },
    "장루 정맥류": { "description": "혈관이 확장된 '장루 정맥류'가 의심됩니다.", "advice": "절대 문지르지 마십시오. 작은 자극에도 대량 출혈이 발생할 수 있습니다.", "emergency_alert": "출혈이 시작되어 멈추지 않으면 즉시 응급실로 가야 합니다." },
    "장루 탈출": { "description": "장루가 평소보다 길게 튀어나온 '장루 탈출'이 의심됩니다.", "advice": "누워서 휴식을 취하면 들어갈 수 있습니다. 복압이 오르는 운동을 피하고 넉넉한 주머니를 사용하세요.", "emergency_alert": "색이 검게 변하거나 통증이 심하면 즉시 응급실로 가십시오." },
    "장루 함몰": { "description": "피부 안쪽으로 들어간 '함몰 장루'가 의심됩니다.", "advice": "배설물이 새기 쉬워 피부 손상이 유발됩니다. 함몰형 볼록판을 사용하면 도움이 됩니다.", "emergency_alert": None },
    "장루 탈장": { "description": "장루 주변이 불룩하게 솟은 '장루 탈장'이 의심됩니다.", "advice": "활동 시 '장루용 복대'를 착용하여 복부를 지지하십시오. 체중 조절과 변비 예방이 중요합니다.", "emergency_alert": "갑작스러운 극심한 복통이나 구토가 동반되면 '장 감돈'일 수 있습니다. 즉시 응급실로 가십시오." },
    "장루 협착": { "description": "배설구가 좁아지는 '장루 협착'이 의심됩니다.", "advice": "변을 묽게 하기 위해 수분 섭취를 늘리십시오. 구멍을 넓히는 시술이 필요할 수 있으니 병원을 예약하십시오.", "emergency_alert": None },
    "상세 불명의 발적": { "description": "원인을 알 수 없는 붉은 발진이 지속됩니다.", "advice": "정확한 진단과 치료를 위해 가까운 병원(장루 간호사)을 방문하십시오.", "emergency_alert": None },
    "자극성 피부염 [L1]": { "description": "배설물 누수로 인해 피부가 붉어진 상태(L1)입니다.", "advice": "[관리법] 깨끗이 씻고 말린 후, '피부 보호 필름(Skin Barrier Film)'을 발라주세요. 구멍 크기를 장루에 맞게 줄이십시오.", "emergency_alert": None },
    "자극성 피부염 [L2]": { "description": "피부 껍질이 벗겨지고 진물이 나는 상태(L2)입니다.", "advice": "[관리법] '파우더 뿌리기 → 털기 → 필름 바르기' 과정을 2~3회 반복하여 인공 보호막(Crusting)을 만드십시오.", "emergency_alert": None },
    "자극성 피부염 [L3]": { "description": "피부가 깊게 파이거나 궤양이 생긴 상태(L3)입니다.", "advice": "[병원 방문] 자가 치료가 위험할 수 있습니다. 상처가 깊으므로 병원을 방문하여 전문적인 처치를 받으십시오.", "emergency_alert": None },
    "기계적 손상 [L1]": { "description": "장루판 제거 시 자극으로 피부가 붉어진 상태(L1)입니다.", "advice": "[관리법] 장루판 제거 시 반드시 '제거제(Remover)'를 사용하여 부드럽게 떼어내십시오. 보호 필름을 사용하십시오.", "emergency_alert": None },
    "기계적 손상 [L2]": { "description": "피부가 찢어져 진물이 나는 상태(L2)입니다.", "advice": "[관리법] 상처 부위에 파우더를 뿌리고 보호 필름을 덧발라 보호막을 만들어주십시오. 제거제를 꼭 사용하십시오.", "emergency_alert": None },
    "기계적 손상 [L3]": { "description": "피부가 깊게 찢어진 심각한 상처(L3)입니다.", "advice": "[병원 방문] 상처 봉합이나 전문 드레싱이 필요할 수 있습니다. 병원을 방문하십시오.", "emergency_alert": None },
    "짓무름": { "description": "습기로 인해 피부가 하얗게 붓는 '짓무름'이 의심됩니다.", "advice": "핵심은 '건조'입니다. 드라이기(찬바람)로 말리고 파우더와 보호 필름을 사용하여 뽀송하게 만드십시오.", "emergency_alert": None },
    "모낭염": { "description": "털구멍 세균 감염인 '모낭염'이 의심됩니다.", "advice": "항생제 연고나 파우더를 사용하십시오. 면도 시 가위나 전기 클리퍼를 사용하여 자극을 줄이십시오.", "emergency_alert": None },
    "곰팡이 감염": { "description": "곰팡이균에 의한 '진균 감염'이 의심됩니다.", "advice": "일반 파우더는 효과가 없습니다. 항진균제 처방을 위해 약국이나 병원을 방문하십시오.", "emergency_alert": None },
    "알레르기": { "description": "제품 성분에 반응하는 '알레르기 피부염'이 의심됩니다.", "advice": "사용 중인 장루판이나 장루주머니를 변경하십시오. 심하면 스테로이드 로션 처방이 필요합니다.", "emergency_alert": None },
    "건선": { "description": "자가면역 질환인 '장루 주위 건선'이 의심됩니다.", "advice": "전문적인 치료가 필요하므로 병원을 방문하십시오.", "emergency_alert": None },
    "요산 결정": { "description": "소변 성분이 굳은 '요산 결정'이 의심됩니다.", "advice": "식초와 물을 1:1로 섞어 거즈에 적셔 10분간 올려두면 녹습니다. 이후 물로 씻어내세요.", "emergency_alert": None },
    "육아종": { "description": "붉은 살점이 덧자란 '육아종'이 의심됩니다.", "advice": "출혈이 잦으니 문지르지 마시고 피부를 말려주세요. 심하면 병원에서 제거해야 합니다.", "emergency_alert": None },
    "기성 사마귀 병변": { "description": "만성 자극으로 피부가 두꺼워진 '기성 사마귀 병변'이 의심됩니다.", "advice": "지속적인 배설물 노출이 원인입니다. 장루판 구멍을 장루 크기에 딱 맞게 줄이십시오.", "emergency_alert": None },
    "점막피부 분리": { "description": "장루와 피부 사이가 벌어진 상태입니다.", "advice": "틈새를 장루 파우더나 연고로 메워서 평평하게 만든 후 장루판을 붙이십시오.", "emergency_alert": None },
    "단순 타박상": { "description": "외부 충격에 의한 '단순 타박상'으로 보입니다.", "advice": "특별한 치료 없이 자연 치유됩니다. 장루판 교체 시 부드럽게 다뤄주세요.", "emergency_alert": None },
    "외상성 궤양": { "description": "외상으로 인해 깊은 상처(궤양)가 발생했습니다.", "advice": "자가 치료가 어려울 수 있습니다. 병원을 방문하여 상처 소독을 받으십시오.", "emergency_alert": None },
    "대장흑색증": { "description": "변비약 복용으로 인한 '대장흑색증'이 의심됩니다.", "advice": "단순 색소 침착으로 건강에 해롭지 않습니다. 특별한 치료가 필요 없습니다.", "emergency_alert": None },
    "붉은 흉터": { "description": "과거 상처의 흔적(흉터)입니다.", "advice": "장루와 피부가 건강합니다. 주기적으로 관리해주세요.", "emergency_alert": None },
    "흉터/색소침착": { "description": "수술이나 과거 상처로 인한 색소침착입니다.", "advice": "현재 문제가 없는 상태입니다. 평소대로 관리하십시오.", "emergency_alert": None },
    "정상": { "description": "장루와 피부가 건강합니다.", "advice": "현재 상태가 아주 좋습니다. 주기적으로 관리해주세요!", "emergency_alert": None }
}



