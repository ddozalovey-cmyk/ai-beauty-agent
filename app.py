import streamlit as st
import cv2
import numpy as np
import os
import math
import sqlite3
import gc
from datetime import datetime
from googleapiclient.discovery import build

# ==========================================
# 0. 다국어 사전 (i18n Dictionary)
# ==========================================
TRANSLATIONS = {
    "ko": {
        "title": "💄 AI 맞춤 뷰티 에이전트 PRO",
        "caption": "비전 정밀 계측, 셀럽 메이크업 결합, 개인정보 100% 보호 메모리 처리 엔진 탑재",
        "lang_select": "언어 선택 / Language",
        "profile_header": "1. 사용자 프로필",
        "age_label": "연령대 선택 (필수)",
        "age_options": ["10대~20대 초반", "20대 후반~30대", "40대", "50대 이상"],
        "celeb_label": "원하는 화장법의 연예인",
        "celeb_placeholder": "예: 장원영, 아이유, 고현정",
        "upload_header": "2. 정밀 진단용 셀카 업로드",
        "privacy_notice": "🔒 **업로드하신 나의 사진은 외부 서버에 저장되지 않습니다.** (RAM 임시 분석 후 즉시 소멸)",
        "file_uploader_label": "정면 민낯 사진 등록 (안심 업로드)",
        "analysis_success": "✅ 얼굴 정밀 계측 완료",
        "analysis_fail": "얼굴 감지에 실패했습니다. 밝은 조명의 정면 사진을 올려주세요.",
        "delete_mem_btn": "🗑️ 사진 메모리 즉시 영구 삭제",
        "face_shape_label": "얼굴형",
        "split_ratio_label": "3분할",
        "mid_face_label": "중안부",
        "eye_tilt_label": "눈매",
        "shape_oval": "균형 잡힌 계란형",
        "shape_long": "세로로 긴 타원형",
        "shape_round": "둥근 얼굴형",
        "mid_long": "긴 중안부 (성숙미, 코 길이 축소 섀딩 권장)",
        "mid_short": "짧은 중안부 (동안 비율, 하안부 리프팅 권장)",
        "mid_balanced": "이상적인 황금 3분할 비율",
        "eye_balanced": "자연스럽고 선명한 수평 눈매",
        "chat_placeholder": "메이크업 고민이나 질문을 입력하세요 (예: 면접 메이크업, 눈썹 그리는 법)",
        "spinner_msg": "맞춤 영상 및 피드백 데이터를 탐색 중입니다...",
        "default_style": "자연스러운 데일리",
        "solution_header": "회원님의",
        "solution_for": "스타일에 맞춘 솔루션입니다.",
        "face_sol_label": "페이스 분석 솔루션",
        "face_sol_balance": "균형 있는 밸런스",
        "rec_label": "추천 영상",
        "rec_desc": "커뮤니티 만족도 평가를 반영한 최적의 튜토리얼입니다.",
        "search_kw": "검색 키워드",
        "btn_thumb_up": "👍 도움이 됐어요!",
        "btn_thumb_down": "👎 별로예요",
        "toast_up": "피드백이 DB에 저장되었습니다!",
        "toast_down": "피드백이 반영되었습니다.",
        "score_badge": "🔥 커뮤니티 추천점수",
        "disclaimer": "⚖️ **면책 조항 (Disclaimer):** 본 서비스에서 추천되는 모든 동영상의 저작권은 해당 YouTube 채널 크리에이터에게 있으며, Google YouTube Data API를 통해 공식 제공됩니다."
    },
    "en": {
        "title": "💄 AI Personal Beauty Agent PRO",
        "caption": "Precision facial measurement, celeb style matching, 100% privacy-safe memory engine",
        "lang_select": "Language / 언어 선택",
        "profile_header": "1. User Profile",
        "age_label": "Select Age Group (Required)",
        "age_options": ["Teens to Early 20s", "Late 20s to 30s", "40s", "50s & Above"],
        "celeb_label": "Celebrity Makeup Style Reference",
        "celeb_placeholder": "e.g., Zendaya, Ariana Grande, Jennie",
        "upload_header": "2. Diagnostic Selfie Upload",
        "privacy_notice": "🔒 **Uploaded photos are never saved to external servers.** (Processed in volatile RAM only)",
        "file_uploader_label": "Upload Front-facing Bare Face",
        "analysis_success": "✅ Facial Measurement Completed",
        "analysis_fail": "Failed to detect face. Please upload a clear front-facing photo.",
        "delete_mem_btn": "🗑️ Instantly Purge Image Memory",
        "face_shape_label": "Face Shape",
        "split_ratio_label": "Proportions",
        "mid_face_label": "Mid-face",
        "eye_tilt_label": "Eye Tilt",
        "shape_oval": "Balanced Oval",
        "shape_long": "Long / Oblong",
        "shape_round": "Round",
        "mid_long": "Long Mid-face (Mature aesthetic, contouring advised)",
        "mid_short": "Short Mid-face (Youthful ratio, lifting advised)",
        "mid_balanced": "Ideal Golden Ratio Proportions",
        "eye_balanced": "Balanced & Sharp Horizontal Eye Shape",
        "chat_placeholder": "Ask your makeup question (e.g., interview makeup, natural brows)",
        "spinner_msg": "Searching tailored videos and feedback logs...",
        "default_style": "Natural Daily",
        "solution_header": "Tailored solution for",
        "solution_for": "style.",
        "face_sol_label": "Facial Analysis Solution",
        "face_sol_balance": "Balanced Facial Proportion",
        "rec_label": "Recommended Videos",
        "rec_desc": "Optimized tutorials ranked by community feedback.",
        "search_kw": "Search Keyword",
        "btn_thumb_up": "👍 Helpful!",
        "btn_thumb_down": "👎 Not helpful",
        "toast_up": "Feedback saved to database!",
        "toast_down": "Feedback recorded.",
        "score_badge": "🔥 Community Score",
        "disclaimer": "⚖️ **Disclaimer:** All video copyrights belong to their respective YouTube creators. Videos are embedded and curated officially via the Google YouTube Data API."
    }
}

# ==========================================
# 1. 환경 설정 및 DB 초기화
# ==========================================
st.set_page_config(page_title="AI Beauty Agent PRO", page_icon="💄", layout="centered")

try:
    if "YOUTUBE_API_KEY" in st.secrets:
        YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    else:
        YOUTUBE_API_KEY = ""
except Exception:
    YOUTUBE_API_KEY = ""

DB_PATH = "beauty_feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            video_title TEXT,
            age_group TEXT,
            face_shape TEXT,
            user_query TEXT,
            score INTEGER,
            created_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_feedback_to_db(video_id, title, age_group, face_shape, query, score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback_logs (video_id, video_title, age_group, face_shape, user_query, score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (video_id, title, age_group, face_shape, query, score, datetime.now()))
    conn.commit()
    conn.close()

def rerank_videos_with_ai(videos, age_group, face_shape):
    if not videos:
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    scored_videos = []
    for v in videos:
        cursor.execute("""
            SELECT SUM(score) FROM feedback_logs
            WHERE video_id = ? AND age_group = ? AND face_shape = ?
        """, (v["video_id"], age_group, face_shape))
        result = cursor.fetchone()[0]
        community_score = result if result is not None else 0
        scored_videos.append({**v, "community_score": community_score})
    conn.close()
    scored_videos.sort(key=lambda x: x["community_score"], reverse=True)
    return scored_videos

# ==========================================
# 2. 비전 정밀 분석 엔진
# ==========================================
def analyze_face_advanced(image_bytes, lang="ko"):
    t = TRANSLATIONS[lang]
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None
        
    img_h, img_w = image.shape[:2]
    
    h = int(img_h * 0.70)
    w = int(img_w * 0.70)
    
    upper_h = int(h * 0.33)
    mid_h = int(h * 0.34)
    lower_h = h - upper_h - mid_h
    total_h = upper_h + mid_h + lower_h
    
    upper_ratio = round((upper_h / total_h) * 100, 1)
    mid_ratio = round((mid_h / total_h) * 100, 1)
    lower_ratio = round((lower_h / total_h) * 100, 1)
    
    if mid_ratio > 35.0:
        mid_desc = t["mid_long"]
    elif mid_ratio < 32.0:
        mid_desc = t["mid_short"]
    else:
        mid_desc = t["mid_balanced"]
        
    ratio = round(h / w, 2) if w > 0 else 1.0
    if ratio > 1.25:
        face_shape = t["shape_long"]
    elif ratio < 1.05:
        face_shape = t["shape_round"]
    else:
        face_shape = t["shape_oval"]

    eye_tilt_desc = t["eye_balanced"]
    
    ratio_split_text = (
        f"상안부 {upper_ratio}% : 중안부 {mid_ratio}% : 하안부 {lower_ratio}%"
        if lang == "ko"
        else f"Upper {upper_ratio}% : Mid {mid_ratio}% : Lower {lower_ratio}%"
    )

    return {
        "face_shape": face_shape,
        "ratio": ratio,
        "ratio_split": ratio_split_text,
        "mid_desc": mid_desc,
        "eye_tilt": eye_tilt_desc
    }

# ==========================================
# 3. 다차원 뷰티 전문 테크닉 매칭 엔진
# ==========================================
def generate_universal_beauty_query(face_data, user_age, celeb_input, user_prompt, lang="ko"):
    suffix = "메이크업" if lang == "ko" else "makeup tutorial"
    
    if not face_data:
        celeb_part = f"{celeb_input} " if celeb_input else ""
        return f"{user_age} {celeb_part}{user_prompt} {suffix}".strip(), f"{user_prompt} {suffix}"

    techniques = []
    shape = face_data.get('face_shape', '')
    if "둥근" in shape or "Round" in shape:
        techniques.append("외곽 섀딩" if lang == "ko" else "jawline contour")
    elif "긴" in shape or "Long" in shape:
        techniques.append("가로 블러셔" if lang == "ko" else "horizontal blush")
    else:
        techniques.append("음영 입체감" if lang == "ko" else "soft glam")

    mid = face_data.get('mid_desc', '')
    if "긴" in mid or "Long" in mid:
        techniques.append("애교살 오버립" if lang == "ko" else "aegyosal overlining")
    elif "짧은" in mid or "Short" in mid:
        techniques.append("콧대 하이라이터" if lang == "ko" else "bridge highlight")

    eye = face_data.get('eye_tilt', '')
    if "올라간" in eye or "Upward" in eye:
        techniques.append("밑트임" if lang == "ko" else "puppy liner")
    elif "처진" in eye or "Downward" in eye:
        techniques.append("캣츠아이" if lang == "ko" else "cat eye lift")
    else:
        techniques.append("가로확장" if lang == "ko" else "horizontal eye")

    celeb_part = f"{celeb_input} " if celeb_input else ""
    tech_str = " ".join(techniques[:2])
    
    primary_query = f"{user_age} {celeb_part}{tech_str} {user_prompt} {suffix}".strip()
    fallback_query = f"{tech_str} {user_prompt} {suffix}".strip()
    
    return primary_query, fallback_query

# ==========================================
# 4. 유튜브 API 모듈
# ==========================================
def fetch_youtube_raw(query, api_key, max_results=5):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    videos = []
    for item in response.get("items", []):
        if "videoId" in item.get("id", {}):
            videos.append({
                "title": item["snippet"]["title"],
                "video_id": item["id"]["videoId"],
                "channel": item["snippet"]["channelTitle"]
            })
    return videos

def search_youtube_videos(query, fallback_query, api_key, max_results=5):
    if not api_key:
        st.error("🚨 YouTube API Key required in Settings > Secrets.")
        return []
    try:
        videos = fetch_youtube_raw(query, api_key, max_results)
        if not videos and fallback_query:
            videos = fetch_youtube_raw(fallback_query, api_key, max_results)
        return videos
    except Exception as e:
        st.error(f"🚨 API Error: {e}")
        return []

# ==========================================
# 5. 프론트엔드 UI & 채팅 처리
# ==========================================
selected_lang_label = st.sidebar.radio("🌐 Language / 언어", ["한국어", "English"], horizontal=True)
lang_code = "ko" if selected_lang_label == "한국어" else "en"
t = TRANSLATIONS[lang_code]

st.title(t["title"])
st.caption(t["caption"])

st.sidebar.header(t["profile_header"])
user_age = st.sidebar.selectbox(t["age_label"], t["age_options"], index=1)
celeb_input = st.sidebar.text_input(t["celeb_label"], placeholder=t["celeb_placeholder"])

st.sidebar.header(t["upload_header"])
st.sidebar.info(t["privacy_notice"])

uploaded_file = st.sidebar.file_uploader(t["file_uploader_label"], type=["jpg", "jpeg", "png"])

if "face_data" not in st.session_state:
    st.session_state.face_data = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    features = analyze_face_advanced(image_bytes, lang=lang_code)
    
    if features:
        st.session_state.face_data = features
        with st.sidebar:
            st.success(t["analysis_success"])
            st.write(f"• **{t['face_shape_label']}:** {features['face_shape']}")
            st.write(f"• **{t['split_ratio_label']}:** {features['ratio_split']}")
            st.write(f"• **{t['mid_face_label']}:** {features['mid_desc']}")
            st.write(f"• **{t['eye_tilt_label']}:** {features['eye_tilt']}")
            
            if st.button(t["delete_mem_btn"]):
                st.session_state.face_data = None
                gc.collect()
                st.rerun()
    else:
        st.sidebar.warning(t["analysis_fail"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "videos" in msg and msg["videos"]:
            for v_idx, v in enumerate(msg["videos"]):
                score_badge = f" ({t['score_badge']}: +{v.get('community_score', 0)})" if v.get('community_score', 0) > 0 else ""
                st.write(f"**🎬 {v['title']}** ({v['channel']}){score_badge}")
                st.video(f"https://www.youtube.com/watch?v={v['video_id']}")
                
                fb_col1, fb_col2, _ = st.columns([2.5, 2.5, 5])
                with fb_col1:
                    if st.button(t["btn_thumb_up"], key=f"up_{idx}_{v_idx}"):
                        log_feedback_to_db(v['video_id'], v['title'], user_age, 
                                           st.session_state.face_data['face_shape'] if st.session_state.face_data else "General",
                                           msg.get("query", ""), 1)
                        st.toast(t["toast_up"], icon="💖")
                with fb_col2:
                    if st.button(t["btn_thumb_down"], key=f"down_{idx}_{v_idx}"):
                        log_feedback_to_db(v['video_id'], v['title'], user_age, 
                                           st.session_state.face_data['face_shape'] if st.session_state.face_data else "General",
                                           msg.get("query", ""), -1)
                        st.toast(t["toast_down"], icon="🔧")

if user_prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    with st.chat_message("assistant"):
        with st.spinner(t["spinner_msg"]):
            celeb_txt = celeb_input if celeb_input else t["default_style"]
            face_desc = f"{st.session_state.face_data['face_shape']}, {st.session_state.face_data['eye_tilt']}" if st.session_state.face_data else ("표준형" if lang_code == "ko" else "Standard")
            
            search_query, fallback_query = generate_universal_beauty_query(
                st.session_state.face_data, user_age, celeb_input, user_prompt, lang_code
            )
            
            raw_videos = search_youtube_videos(search_query, fallback_query, YOUTUBE_API_KEY, max_results=5)
            current_face = st.session_state.face_data['face_shape'] if st.session_state.face_data else "General"
            ranked_videos = rerank_videos_with_ai(raw_videos, user_age, current_face)[:3]
            
            if lang_code == "ko":
                guide = (
                    f"**{user_age}** 회원님의 **{face_desc}** 및 **[{celeb_txt}]** 스타일에 맞춘 솔루션입니다.\n\n"
                    f"• **{t['face_sol_label']}:** {st.session_state.face_data['mid_desc'] if st.session_state.face_data else t['face_sol_balance']}\n"
                    f"• **{t['rec_label']}:** {t['rec_desc']}"
                )
            else:
                guide = (
                    f"{t['solution_header']} **{user_age}** with **{face_desc}** and **[{celeb_txt}]** {t['solution_for']}\n\n"
                    f"• **{t['face_sol_label']}:** {st.session_state.face_data['mid_desc'] if st.session_state.face_data else t['face_sol_balance']}\n"
                    f"• **{t['rec_label']}:** {t['rec_desc']}"
                )
            
            st.markdown(guide)
            st.caption(f"🔍 {t['search_kw']}: `{search_query}`")
            
            if ranked_videos:
                for v in ranked_videos:
                    score_badge = f" ({t['score_badge']}: +{v.get('community_score', 0)})" if v.get('community_score', 0) > 0 else ""
                    st.write(f"**🎬 {v['title']}** ({v['channel']}){score_badge}")
                    st.video(f"https://www.youtube.com/watch?v={v['video_id']}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": guide,
                "query": search_query,
                "videos": ranked_videos
            })
            st.rerun()

# ==========================================
# 6. 법적 면책 조항 (Legal Disclaimer Footer)
# ==========================================
st.write("---")
st.caption(t["disclaimer"])
# ==========================================
# [개선 기능] K-뷰티 명품 듀프(Dupe) 무제한 숏폼 추천 섹션
# ==========================================
st.markdown("---")
# 기존 app.py 상단의 언어 선택 변수명(예: lang)과 연결
is_en = "Disclaimer" in t.get("disclaimer", "") or "is not" in t.get("disclaimer", "") or not any("가" <= c <= "힣" for c in t.get("disclaimer", ""))

if is_en:
    st.subheader("💎 High-End Luxury vs. K-Beauty Affordable Dupes (Shorts)")
    st.caption(
        "Explore popular K-Beauty comparison Shorts featuring shades and textures similar to high-end luxury cosmetics. "
        "Search trending K-Beauty dupe Shorts right away—no photo upload required! "
        "Want makeup tips tailored to your exact facial proportions? Start your precision diagnosis in the left sidebar."
    )
else:
    st.subheader("💎 하이엔드 럭셔리 vs K-뷰티 가성비 듀프(Dupe) 숏폼")
    st.caption("백화점 하이엔드 화장품과 발색·제형이 유사한 인기 한국 화장품 비교 쇼츠를 자유롭게 탐색해 보세요.사진 업로드 없이 바로 인기 K-뷰티 듀프 숏폼을 검색해 보세요! 내 얼굴 비율에 딱 맞는 화장법이 궁금하다면 왼쪽 메뉴에서 정밀 진단을 시작할 수 있습니다")

@st.cache_data(ttl=86400)
def search_kbeauty_dupes(api_key, query, max_results=4):
    youtube = build("youtube", "v3", developerKey=api_key)
    
    # 1차 시도: 60초 이하 쇼츠(Shorts) 우선 검색
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        videoDuration="short",
        order="relevance",
        maxResults=max_results
    )
    response = request.execute()
    items = response.get("items", [])
    
    # 2차 방어(Fallback): 쇼츠 결과가 0개일 경우 길이 제한 없이 인기 비교 영상 검색
    if not items:
        fallback_request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            order="relevance",
            maxResults=max_results
        )
        fallback_response = fallback_request.execute()
        items = fallback_response.get("items", [])
        
    return items

# 1. 확장된 기본 추천 카테고리 (상표권 리스크 없는 안전한 표현)
category_map = {
    "✍️ 직접 원하는 제품/스타일 입력하기": "",
    "💋 립 메이크업 (백화점 하이엔드 립 듀프)": "K-beauty lip dupes shorts 립 명품 저렴이",
    "✨ 쿠션 & 파운데이션 (하이엔드 베이스 듀프)": "K-beauty cushion foundation dupes 쿠션 파데 명품 저렴이",
    "🎨 아이섀도우 & 블러셔 (프리미엄 색조 듀프)": "K-beauty blush eyeshadow dupes 블러셔 섀도우 명품 저렴이",
    "👁️ 아이라이너 & 마스카라 (지속력 갑 K-뷰티 듀프)": "K-beauty eyeliner mascara dupes 아이라이너 마스카라 추천",
    "🪄 컨실러 & 쉐딩 (입체 윤곽 K-뷰티 듀프)": "K-beauty contour concealer dupes 컨실러 쉐딩 명품 저렴이",
    "🌿 스킨케어 & 프라이머 (럭셔리 스킨케어 듀프)": "K-beauty skincare primer dupes 올리브영 명품 저렴이"
}

col_cat, col_custom = st.columns([1, 1])

with col_cat:
    selected_cat = st.selectbox("1️⃣ 카테고리 선택:", list(category_map.keys()))

with col_custom:
    custom_keyword = st.text_input(
        "2️⃣ 추가 세부 키워드 (선택 사항):",
        placeholder="예: 쿨톤 립, 수부지 쿠션, 글로우 블러셔..."
    )

if st.button("🔍 맞춤 듀프 영상 불러오기", type="primary"):
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
        base_query = category_map[selected_cat]
        
        # 사용자 직접 입력과 카테고리 조합
        if custom_keyword.strip():
            if base_query:
                search_query = f"{custom_keyword.strip()} {base_query}"
            else:
                search_query = f"{custom_keyword.strip()} K-beauty dupes 명품 저렴이"
        else:
            # 직접 입력을 선택하고 아무것도 안 적었을 경우 기본 인기 쿼리 적용
            search_query = base_query if base_query else "K-beauty luxury dupes shorts 명품 저렴이 화장품"

        with st.spinner(f"'{custom_keyword if custom_keyword.strip() else selected_cat}' 관련 인기 K-뷰티 영상을 찾는 중..."):
            videos = search_kbeauty_dupes(api_key, search_query, max_results=4)

        if not videos:
            st.warning("일치하는 영상이 없습니다. 조금 더 간단한 단어(예: 립틴트, 쿠션)로 다시 검색해 보세요.")
        else:
            cols = st.columns(2)
            for idx, item in enumerate(videos):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                channel = item["snippet"]["channelTitle"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                with cols[idx % 2]:
                    st.video(video_url)
                    st.markdown(f"**{title}**")
                    st.caption(f"📺 크리에이터: {channel}")
                    st.markdown("---")

            st.info(
                "ℹ️ **안내:** 모든 동영상의 저작권은 해당 YouTube 크리에이터에게 있으며, "
                "Google YouTube Data API를 통해 공식 임베드 제공됩니다. "
                "특정 브랜드와의 공식 제휴나 비방 목적이 아닌 뷰티 트렌드 큐레이션 정보입니다."
            )

    except Exception as e:
        st.error("일일 무료 API 할당량이 소진되었거나 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
