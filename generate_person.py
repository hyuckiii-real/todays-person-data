import os
import json
import time
from datetime import datetime, timezone, timedelta
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. 한국 표준시(KST) 기준 오늘 날짜 및 요일 계산
KST = timezone(timedelta(hours=9))
now_kst = datetime.now(KST)
today_str = now_kst.strftime("%Y-%m-%d")
month_day = f"{now_kst.month}월 {now_kst.day}일"
weekday_idx = now_kst.weekday()  # 0:월 ~ 6:일

# 2. 요일별 강제 테마 로테이션
WEEKLY_THEMES = {
    0: "과학 / IT / 발명",
    1: "예술 / 문학 / 음악",
    2: "역사 / 철학 / 사상",
    3: "비즈니스 / 경제 / 혁신",
    4: "한국의 인물 (역사·문화·과학 전반)",
    5: "의외의 인물 / 모험가 / 탐험가",
    6: "의외의 인물 / 스포츠·사회공헌·숨은 영웅",
}
today_theme = WEEKLY_THEMES[weekday_idx]

# 3. 기존 archive.json 로드 (중복 방지용)
ARCHIVE_FILE = "archive.json"
TODAY_FILE = "today.json"

if os.path.exists(ARCHIVE_FILE):
    with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
        archive_data = json.load(f)
else:
    archive_data = []

# 이미 같은 날짜 데이터가 있으면 중복 생성 방지
if archive_data and archive_data[-1].get("date") == today_str:
    print(f"[{today_str}] 이미 오늘의 인물이 생성되어 있습니다.")
    exit(0)

excluded_names = [item["name"] for item in archive_data[-365:]]
excluded_str = ", ".join(excluded_names) if excluded_names else "없음"

# 4. 출력 JSON 스키마 엄격 정의
class PersonOfTheDay(BaseModel):
    date: str = Field(description="오늘 날짜 (YYYY-MM-DD)")
    theme: str = Field(description="오늘의 요일 테마")
    name: str = Field(description="인물 이름 (예: 앨런 튜링 (Alan Turing) 또는 세종대왕)")
    lifespan: str = Field(description="생몰연도 (예: 1912 - 1954 또는 1397 - 1450)")
    category: str = Field(description="세부 분야 (예: 컴퓨터공학 / 수학)")
    one_liner: str = Field(description="이 인물을 설명하는 강렬한 한 줄 수식어 (25자 이내)")
    why_today: str = Field(description=f"{month_day} 또는 오늘 선정된 이유와 시의성 설명 (1~2문장)")
    achievements: list[str] = Field(description="핵심 업적 3가지 (각 1문장 이내, 개조식)")
    quote_original: str = Field(description="실제 기록에 남은 검증된 대표 어록 원문 (한국인은 한글, 외국인은 영어/원문. 검증된 명언이 없으면 빈 문자열)")
    quote_ko: str = Field(description="대표 어록의 한국어 번역 또는 검증된 명언이 없을 경우 인물의 핵심 철학을 요약한 한 문장")
    tmi: str = Field(description="교과서 외적인 인간적인 면모나 흥미로운 검증된 일화 (2문장 이내)")

# 5. 프롬프트
prompt = f"""
오늘은 {today_str} ({month_day})이며, 오늘의 필수 테마는 **[{today_theme}]**입니다.
이 테마에 부합하는 '오늘의 인물' 1명을 선정해 JSON으로 작성해 주세요.

[필수 선정 및 할루시네이션 방어 규칙]
1. **시의성 우선 탐색**: {month_day}에 태어났거나, 세상을 떠났거나, 결정적 업적을 이룬 실존 인물을 최우선으로 찾으세요. 만약 {month_day}에 딱 맞는 인물이 없다면, 억지로 날짜를 끼워 맞추지 말고 오늘의 테마([{today_theme}])에 가장 걸맞은 인물을 선정한 뒤 `why_today`에 선정 의미를 자연스럽게 작성하세요.
2. **중복 절대 금지**: 아래 이미 선정된 인물 목록에 있는 사람은 절대 다시 뽑지 마세요.
   - 제외 목록: [{excluded_str}]
3. **논란 및 리스크 원천 차단**: 범죄 이력, 심각한 윤리적 결함, 현대 정치적 극단주의 논란이 있는 인물은 절대 제외하세요. 가급적 역사적 평가가 완료된 작고 인물 위주로 선정하세요.
4. **팩트 및 생몰연도 엄수**: 생몰연도, 국적, 주요 업적의 연도와 수치는 역사적 사실만 기재하세요. 불확실한 추측은 배제합니다.
5. **가짜 명언 생성 금지**: 인물이 실제로 남긴 검증된 어록만 `quote_original`과 `quote_ko`에 적으세요. 확실한 명언이 없는 인물이라면 `quote_original`은 반드시 빈 문자열("")로 두고, `quote_ko`에 인물의 삶과 철학을 관통하는 요약 문장을 적으세요.
6. **가독성**: 모바일 카드 화면에서 30초~1분 안에 읽히도록 간결하고 흥미로운 한국어 문체로 작성하세요.
"""

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),
)

# 6. 점진적 대기(30초~180초) + 6회 이후 폴백 모델 전환으로 503 과부하 완벽 방어
response = None
max_retries = 8
for attempt in range(max_retries):
    # 앞선 6번은 gemini-3.8-flash 시도, 계속 혼잡하면 마지막 2번은 다른 모델로 우회
    target_model = "gemini-3.8-flash" if attempt < 6 else "gemini-3.5-flash-lite"
    try:
        print(f"시도 {attempt + 1}/{max_retries} (모델: {target_model})...")
        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PersonOfTheDay,
                temperature=0.4,
            ),
        )
        break
    except Exception as e:
        wait_sec = min(30 * (attempt + 1), 180)  # 30초, 60초, 90초, 120초, 150초, 180초...
        print(f"시도 {attempt + 1}/{max_retries} 실패: {e}")
        if attempt < max_retries - 1:
            print(f"{wait_sec}초 대기 후 재시도합니다...")
            time.sleep(wait_sec)
        else:
            raise e

person_data = json.loads(response.text)
person_data["date"] = today_str

# 7. today.json 및 archive.json 저장
with open(TODAY_FILE, "w", encoding="utf-8") as f:
    json.dump(person_data, f, ensure_ascii=False, indent=2)

archive_data.append(person_data)
with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
    json.dump(archive_data, f, ensure_ascii=False, indent=2)

print(f"성공적으로 생성 완료: {person_data['name']} ({today_theme})")
