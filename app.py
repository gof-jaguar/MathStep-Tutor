"""
MathStep Tutor — เครื่องมือฝึกวิเคราะห์โจทย์และสอนวิธีทำทีละขั้นตอน
"""

import json
import os
import re
from typing import Optional
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PIL import Image

# โหลด API Key: st.secrets (Cloud) → .env (Local) → env var
load_dotenv()

def _get_api_key() -> str:
    """Resolve API key from Streamlit secrets, .env, or environment."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GEMINI_API_KEY", "")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MathStep Tutor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# i18n — Bilingual labels
# ──────────────────────────────────────────────
LANG = {
    "TH": {
        "subtitle": "เครื่องมือฝึกวิเคราะห์โจทย์ & สอนวิธีทำทีละขั้นตอน",
        "lang_toggle": "🇬🇧 English",
        "api_title": "🔑 ตั้งค่า API Key",
        "api_placeholder": "วาง API Key ของคุณที่นี่...",
        "api_help": "รับ API Key ฟรีได้ที่ Google AI Studio (aistudio.google.com)",
        "api_save": "✅  บันทึก API Key",
        "api_warn": "กรุณากรอก API Key",
        "sidebar_settings": "⚙️ ตั้งค่า",
        "sidebar_change_key": "🔄 เปลี่ยน API Key",
        "legend_data": "ข้อมูลจากโจทย์",
        "legend_op": "เครื่องหมายคำนวณ",
        "legend_result": "ผลลัพธ์ขั้นตอน",
        "legend_answer": "คำตอบสุดท้าย",
        "input_title": "✏️ ป้อนโจทย์คณิตศาสตร์",
        "input_placeholder": "เช่น: แม่ค้าซื้อส้ม 5 กิโลกรัม กิโลกรัมละ 40 บาท และซื้อแอปเปิ้ล 3 กิโลกรัม กิโลกรัมละ 75 บาท แม่ค้าต้องจ่ายเงินทั้งหมดเท่าไร?",
        "upload_label": "📷 หรืออัปโหลดรูปโจทย์",
        "upload_caption": "รูปโจทย์ที่อัปโหลด",
        "submit": "🚀  วิเคราะห์โจทย์",
        "warn_empty": "กรุณาพิมพ์โจทย์หรืออัปโหลดรูปภาพ",
        "spinner": "🤔 กำลังวิเคราะห์โจทย์...",
        "err_json": "ไม่สามารถอ่านคำตอบจาก AI ได้ กรุณาลองใหม่อีกครั้ง",
        "err_generic": "เกิดข้อผิดพลาด",
        "problem_label": "📝 โจทย์",
        "new_problem": "🔄 โจทย์ใหม่",
        "analysis_title": "🔍 การวิเคราะห์โจทย์",
        "topic_label": "📌 หัวข้อ",
        "given_label": "📥 สิ่งที่โจทย์บอก",
        "find_label": "❓ สิ่งที่โจทย์ถาม",
        "keywords_label": "🔑 คีย์เวิร์ดสำคัญ",
        "logic_label": "🧠 ตรรกะเบื้องหลัง",
        "equation_label": "📝 สมการ",
        "step_label": "ขั้นตอน",
        "next_step": "👉  ดูขั้นตอนที่",
        "all_done": "แสดงครบทุกขั้นตอนแล้ว!",
        "all_done_sub": "ลองทำโจทย์ใหม่เพื่อฝึกฝนเพิ่มเติม",
        "start_new": "✏️  เริ่มโจทย์ใหม่",
        "image_prompt": "\n\nช่วยอ่านโจทย์จากรูปภาพนี้แล้ววิเคราะห์ให้หน่อย",
        "extra_text": "\n\nข้อความเพิ่มเติม: ",
        "image_fallback": "(โจทย์จากรูปภาพ)",
    },
    "EN": {
        "subtitle": "Analyze problems & learn step-by-step solutions",
        "lang_toggle": "🇹🇭 ภาษาไทย",
        "api_title": "🔑 Set API Key",
        "api_placeholder": "Paste your API Key here...",
        "api_help": "Get a free API Key at Google AI Studio (aistudio.google.com)",
        "api_save": "✅  Save API Key",
        "api_warn": "Please enter an API Key",
        "sidebar_settings": "⚙️ Settings",
        "sidebar_change_key": "🔄 Change API Key",
        "legend_data": "Data from problem",
        "legend_op": "Operators",
        "legend_result": "Step result",
        "legend_answer": "Final answer",
        "input_title": "✏️ Enter a Math Problem",
        "input_placeholder": "e.g.: A shopkeeper buys 5 kg of oranges at $2 per kg and 3 kg of apples at $3.50 per kg. How much does she pay in total?",
        "upload_label": "📷 Or upload an image of the problem",
        "upload_caption": "Uploaded problem image",
        "submit": "🚀  Analyze Problem",
        "warn_empty": "Please type a problem or upload an image",
        "spinner": "🤔 Analyzing the problem...",
        "err_json": "Could not parse AI response. Please try again.",
        "err_generic": "Error",
        "problem_label": "📝 Problem",
        "new_problem": "🔄 New Problem",
        "analysis_title": "🔍 Problem Analysis",
        "topic_label": "📌 Topic",
        "given_label": "📥 Given",
        "find_label": "❓ Find",
        "keywords_label": "🔑 Keywords",
        "logic_label": "🧠 Logic Behind",
        "equation_label": "📝 Equation",
        "step_label": "Step",
        "next_step": "👉  Show Step",
        "all_done": "All steps revealed!",
        "all_done_sub": "Try a new problem to keep practicing",
        "start_new": "✏️  Start New Problem",
        "image_prompt": "\n\nPlease read the problem from this image and analyze it.",
        "extra_text": "\n\nAdditional context: ",
        "image_fallback": "(Problem from image)",
    },
}

# ──────────────────────────────────────────────
# System instructions per language
# ──────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = {
    "TH": """คุณคือติวเตอร์อัจฉริยะที่เชี่ยวชาญการสอนวิธีคิด เมื่อได้รับโจทย์ (ไม่ว่าจะเป็นสมการหรือโจทย์ปัญหาภาษาไทยยาวๆ) ให้เน้นอธิบาย 'ตรรกะเบื้องหลัง' ว่าทำไมถึงต้องตั้งสมการแบบนั้น และคีย์เวิร์ดในโจทย์คืออะไร เพื่อให้ผู้ใช้ฝึกทักษะการวิเคราะห์โจทย์ได้ด้วยตนเอง

ตอบกลับเป็น JSON เท่านั้น ตามโครงสร้างนี้:
{
  "topic": "หัวข้อ/ประเภทของโจทย์",
  "analysis": {
    "given": "สิ่งที่โจทย์บอก (ข้อมูลที่ให้มา) — อธิบายสั้นกระชับ",
    "find": "สิ่งที่โจทย์ถาม — อธิบายสั้นกระชับ",
    "keywords": "คีย์เวิร์ดสำคัญในโจทย์ที่บ่งบอกวิธีคิด",
    "logic": "อธิบายตรรกะเบื้องหลังว่าทำไมเราถึงต้องใช้วิธีนี้"
  },
  "equation": "สมการหรือนิพจน์ที่ตั้งขึ้น (ถ้ามี)",
  "steps": [
    {
      "title": "ชื่อขั้นตอนสั้นๆ",
      "explanation": "คำอธิบายวิธีทำ โดยใช้ HTML ดังนี้: ตัวเลขจากโจทย์ใส่ <span style='color:#2E86C1;font-weight:600;'>สีน้ำเงิน</span>, เครื่องหมาย +−×÷ ใส่ <span style='color:#E67E22;font-weight:600;'>สีส้ม</span>, ผลลัพธ์ใส่ <span style='color:#27AE60;font-weight:600;'>สีเขียว</span>, คำตอบสุดท้ายใส่ <span style='color:#E74C3C;font-weight:700;'>สีแดง</span>"
    }
  ]
}

กฎสำคัญ:
- ตอบเป็น JSON เท่านั้น ห้ามมี markdown code fence ครอบ
- ทุกขั้นตอนต้องอธิบายเหตุผล "ทำไม" ไม่ใช่แค่ "ทำอะไร"
- ขั้นตอนสุดท้ายต้องสรุปคำตอบชัดเจน
- ใช้ภาษาไทย อธิบายเข้าใจง่าย เหมือนพี่สอนน้อง
- ถ้าโจทย์เป็นรูปภาพ ให้อ่านโจทย์จากภาพแล้ววิเคราะห์เหมือนกัน""",

    "EN": """You are a brilliant math tutor who specializes in teaching HOW to think. When given a problem (equations or word problems), focus on explaining the 'logic behind' why we set up the equation that way, and what the key clues in the problem are, so the student can develop their own problem-analysis skills.

Reply in JSON only, following this structure:
{
  "topic": "Topic / type of problem",
  "analysis": {
    "given": "What the problem tells us (given data) — concise",
    "find": "What the problem asks — concise",
    "keywords": "Key clues in the problem that hint at the method",
    "logic": "Explain the reasoning behind why we use this approach"
  },
  "equation": "The equation or expression set up (if any)",
  "steps": [
    {
      "title": "Short step title",
      "explanation": "Explanation using HTML colors: numbers from the problem in <span style='color:#2E86C1;font-weight:600;'>blue</span>, operators +−×÷ in <span style='color:#E67E22;font-weight:600;'>orange</span>, intermediate results in <span style='color:#27AE60;font-weight:600;'>green</span>, final answer in <span style='color:#E74C3C;font-weight:700;'>red</span>"
    }
  ]
}

Important rules:
- Reply with JSON only, no markdown code fences
- Every step must explain WHY, not just WHAT
- The last step must clearly state the final answer
- Use simple, friendly English — like a tutor explaining to a younger student
- If the problem is an image, read it from the image and analyze it the same way""",
}

# ──────────────────────────────────────────────
# CSS — Mac + iPad optimised, responsive
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import clean fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=Inter:wght@300;400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Sarabun', sans-serif;
}

/* ── Responsive container ── */
.block-container {
    max-width: 860px;
    margin: 0 auto;
    padding: 1.5rem 2rem 4rem 2rem;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.8rem 0;
}
.app-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #2E86C1, #8E44AD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    color: #888;
    font-size: 1.1rem;
    margin-top: 0.3rem;
}

/* ── Language toggle chip ── */
.lang-bar {
    text-align: center;
    margin-bottom: 0.8rem;
}

/* ── Card ── */
.card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Analysis card ── */
.analysis-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #f8f0ff 100%);
    border: 1px solid #d5d5f5;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.analysis-row {
    margin-bottom: 0.6rem;
    font-size: 1.05rem;
    line-height: 1.7;
}

/* ── Step card ── */
.step-card {
    background: #ffffff;
    border-left: 5px solid #2E86C1;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    animation: fadeSlideIn 0.5s ease-out;
}
.step-card.final {
    border-left-color: #E74C3C;
    background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #2E86C1;
    color: #fff;
    font-weight: 700;
    border-radius: 50%;
    width: 34px; height: 34px;
    margin-right: 0.6rem;
    font-size: 0.95rem;
    flex-shrink: 0;
}
.step-number.final {
    background: #E74C3C;
}

/* ── Equation display ── */
.equation-box {
    background: #fefbe9;
    border: 1px solid #f0e68c;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
    font-size: 1.25rem;
    text-align: center;
    font-weight: 600;
}

/* ── Big touch-friendly buttons ── */
div.stButton > button {
    width: 100%;
    padding: 0.85rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 14px;
    border: none;
    transition: all 0.2s ease;
    min-height: 54px;
    font-family: 'Inter', 'Sarabun', sans-serif;
    cursor: pointer;
}
div.stButton > button:active {
    transform: scale(0.97);
}

/* Primary button */
.primary-btn > button {
    background: linear-gradient(135deg, #2E86C1, #3498DB) !important;
    color: white !important;
}
.primary-btn > button:hover {
    background: linear-gradient(135deg, #2471A3, #2E86C1) !important;
    box-shadow: 0 4px 16px rgba(46,134,193,0.3);
}

/* Next-step button */
.next-btn > button {
    background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
    color: white !important;
    font-size: 1.25rem !important;
    min-height: 62px !important;
}
.next-btn > button:hover {
    box-shadow: 0 4px 16px rgba(39,174,96,0.35);
}

/* Reset button */
.reset-btn > button {
    background: #f5f5f5 !important;
    color: #666 !important;
    border: 1px solid #ddd !important;
}
.reset-btn > button:hover {
    background: #eee !important;
}

/* ── Animation ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeSlideIn 0.5s ease-out;
}

/* ── Textarea ── */
div[data-testid="stTextArea"] textarea {
    font-size: 1.1rem !important;
    font-family: 'Inter', 'Sarabun', sans-serif !important;
    min-height: 130px;
    border-radius: 12px !important;
}

/* ── Color legend ── */
.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 0.8rem 0;
    font-size: 0.95rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.35rem;
}
.legend-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Progress text ── */
.progress-text {
    text-align: center;
    font-size: 1rem;
    color: #888;
    margin-bottom: 0.5rem;
}

/* ── Completion card ── */
.done-card {
    text-align: center;
    background: linear-gradient(135deg, #f0fff0, #fff);
    border: 1px solid #c3e6cb;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    margin-bottom: 1rem;
    animation: fadeSlideIn 0.5s ease-out;
}

/* ── Hide Streamlit chrome ── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* ── Desktop (Mac) tweaks ── */
@media (min-width: 1024px) {
    .block-container {
        max-width: 780px;
        padding: 2rem 2.5rem 4rem 2.5rem;
    }
    .app-header h1 { font-size: 2.6rem; }
    .step-card { padding: 1.5rem 1.8rem; }
    .analysis-card { padding: 1.8rem 2rem; }
    div.stButton > button { min-height: 50px; font-size: 1.05rem; }
    .next-btn > button { min-height: 58px !important; font-size: 1.15rem !important; }
}

/* ── Tablet / iPad ── */
@media (min-width: 768px) and (max-width: 1023px) {
    .block-container {
        max-width: 720px;
        padding: 1.5rem 1.5rem 4rem 1.5rem;
    }
    div.stButton > button { min-height: 58px; }
    .next-btn > button { min-height: 66px !important; font-size: 1.3rem !important; }
}

/* ── Mobile / iPad mini portrait ── */
@media (max-width: 767px) {
    .block-container { padding: 1rem 0.8rem 4rem 0.8rem; }
    .app-header h1 { font-size: 1.8rem; }
    .app-header p { font-size: 0.95rem; }
    .card, .analysis-card, .step-card { padding: 1.2rem 1rem; border-radius: 12px; }
    div.stButton > button { min-height: 56px; font-size: 1.1rem; }
    .next-btn > button { min-height: 64px !important; font-size: 1.25rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────
DEFAULTS = {
    "api_key": _get_api_key(),
    "api_key_set": bool(_get_api_key()),
    "problem_text": "",
    "uploaded_image": None,
    "ai_result": None,
    "visible_steps": 0,
    "is_loading": False,
    "lang": "TH",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def t(key: str) -> str:
    """Get translated string for current language."""
    return LANG[st.session_state.lang].get(key, key)


# ──────────────────────────────────────────────
# Helper: call Gemini
# ──────────────────────────────────────────────
def call_gemini(problem_text: str, image: Optional[Image.Image] = None) -> Optional[dict]:
    """Send the problem to Gemini and return parsed JSON dict."""
    genai.configure(api_key=st.session_state.api_key)

    lang = st.session_state.lang
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTIONS[lang],
    )

    parts = []
    if image is not None:
        parts.append(image)
        if problem_text.strip():
            parts.append(t("extra_text") + problem_text)
        else:
            parts.append(t("image_prompt"))
    else:
        parts.append(problem_text)

    response = model.generate_content(parts)
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ──────────────────────────────────────────────
# Render helpers
# ──────────────────────────────────────────────
def render_header():
    st.markdown(
        f"""
    <div class="app-header">
        <h1>MathStep Tutor</h1>
        <p>{t("subtitle")}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_legend():
    st.markdown(
        f"""
    <div class="legend">
        <div class="legend-item"><span class="legend-dot" style="background:#2E86C1;"></span> {t("legend_data")}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#E67E22;"></span> {t("legend_op")}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#27AE60;"></span> {t("legend_result")}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#E74C3C;"></span> {t("legend_answer")}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_analysis(data: dict):
    a = data.get("analysis", {})
    st.markdown(
        f"""
    <div class="analysis-card animate-in">
        <div class="card-title">{t("analysis_title")}</div>
        <div class="analysis-row"><strong>{t("topic_label")}:</strong> {data.get("topic", "-")}</div>
        <div class="analysis-row"><strong>{t("given_label")}:</strong> {a.get("given", "-")}</div>
        <div class="analysis-row"><strong>{t("find_label")}:</strong> {a.get("find", "-")}</div>
        <div class="analysis-row"><strong>{t("keywords_label")}:</strong> {a.get("keywords", "-")}</div>
        <div class="analysis-row"><strong>{t("logic_label")}:</strong> {a.get("logic", "-")}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_equation(eq: str):
    if eq and eq.strip() and eq.strip() != "-":
        st.markdown(
            f"""
        <div class="equation-box animate-in">
            {t("equation_label")}: {eq}
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_step(step: dict, index: int, is_last: bool):
    num_class = "step-number final" if is_last else "step-number"
    card_class = "step-card final animate-in" if is_last else "step-card animate-in"
    st.markdown(
        f"""
    <div class="{card_class}">
        <div style="display:flex;align-items:flex-start;gap:0.4rem;margin-bottom:0.5rem;">
            <span class="{num_class}">{index + 1}</span>
            <strong style="font-size:1.1rem;line-height:34px;">{step.get("title", "")}</strong>
        </div>
        <div style="font-size:1.08rem;line-height:1.85;">
            {step.get("explanation", "")}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def reset_session():
    """Clear solution data and reset to input mode."""
    st.session_state.ai_result = None
    st.session_state.visible_steps = 0
    st.session_state.problem_text = ""
    st.session_state.uploaded_image = None
    st.session_state.is_loading = False


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
render_header()

# ── Language toggle (always visible) ──
col_lang_l, col_lang_m, col_lang_r = st.columns([2, 1, 2])
with col_lang_m:
    if st.button(t("lang_toggle"), use_container_width=True):
        st.session_state.lang = "EN" if st.session_state.lang == "TH" else "TH"
        # Reset AI result when switching language so prompts match
        if st.session_state.ai_result is not None:
            reset_session()
        st.rerun()

st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

# ── API Key Section ──
if not st.session_state.api_key_set:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card-title">{t("api_title")}</div>', unsafe_allow_html=True
    )
    key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder=t("api_placeholder"),
        help=t("api_help"),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_key = st.columns([1, 1])
    with col_key[1]:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button(t("api_save"), use_container_width=True):
            if key_input.strip():
                st.session_state.api_key = key_input.strip()
                st.session_state.api_key_set = True
                st.rerun()
            else:
                st.warning(t("api_warn"))
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"### {t('sidebar_settings')}")
    if st.button(t("sidebar_change_key")):
        st.session_state.api_key_set = False
        st.session_state.api_key = ""
        st.rerun()
    st.divider()
    st.markdown(
        f"""
    <div class="legend" style="flex-direction:column;gap:0.6rem;">
        <div class="legend-item"><span class="legend-dot" style="background:#2E86C1;"></span> {t("legend_data")}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#E67E22;"></span> {t("legend_op")}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#27AE60;"></span> {t("legend_result")}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#E74C3C;"></span> {t("legend_answer")}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────
# INPUT MODE
# ──────────────────────────────────────────
if st.session_state.ai_result is None:
    st.markdown(
        f"""
    <div class="card">
        <div class="card-title">{t("input_title")}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    problem = st.text_area(
        t("input_title"),
        placeholder=t("input_placeholder"),
        height=150,
        label_visibility="collapsed",
    )

    st.markdown(
        f"<p style='font-size:1.05rem;margin-top:0.5rem;'>{t('upload_label')}</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        t("upload_label"),
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption=t("upload_caption"), use_container_width=True)
        st.session_state.uploaded_image = img
    else:
        st.session_state.uploaded_image = None

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    submit = st.button(t("submit"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        has_text = problem and problem.strip()
        has_image = st.session_state.uploaded_image is not None
        if not has_text and not has_image:
            st.warning(t("warn_empty"))
        else:
            with st.spinner(t("spinner")):
                try:
                    result = call_gemini(
                        problem or "", st.session_state.uploaded_image
                    )
                    st.session_state.ai_result = result
                    st.session_state.visible_steps = 0
                    st.session_state.problem_text = problem or t("image_fallback")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error(t("err_json"))
                except Exception as e:
                    st.error(f"{t('err_generic')}: {e}")

# ──────────────────────────────────────────
# RESULT MODE
# ──────────────────────────────────────────
else:
    data = st.session_state.ai_result
    steps = data.get("steps", [])
    total_steps = len(steps)
    visible = st.session_state.visible_steps

    # Top bar
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        display_text = st.session_state.problem_text[:80]
        ellipsis = "..." if len(st.session_state.problem_text) > 80 else ""
        st.markdown(
            f"<div style='font-size:1.05rem;color:#888;padding:0.4rem 0;'>{t('problem_label')}: {display_text}{ellipsis}</div>",
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button(t("new_problem"), use_container_width=True):
            reset_session()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    render_legend()
    render_analysis(data)
    render_equation(data.get("equation", ""))

    # Progress
    if total_steps > 0:
        progress_pct = min(visible / total_steps, 1.0)
        st.markdown(
            f"<div class='progress-text'>{t('step_label')} {min(visible, total_steps)} / {total_steps}</div>",
            unsafe_allow_html=True,
        )
        st.progress(progress_pct)

    # Revealed steps
    for i in range(min(visible, total_steps)):
        is_last = i == total_steps - 1 and visible >= total_steps
        render_step(steps[i], i, is_last)

    # Next or done
    if visible < total_steps:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="next-btn">', unsafe_allow_html=True)
        if st.button(
            f"{t('next_step')} {visible + 1}",
            use_container_width=True,
        ):
            st.session_state.visible_steps += 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
        <div class="done-card">
            <div style="font-size:1.8rem;margin-bottom:0.4rem;">🎉</div>
            <div style="font-size:1.2rem;font-weight:600;color:#27AE60;">{t("all_done")}</div>
            <div style="font-size:1rem;color:#888;margin-top:0.3rem;">{t("all_done_sub")}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button(t("start_new"), use_container_width=True):
            reset_session()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
