# ==============================
# SENTISCOPE — AI SENTIMENT ANALYZER
# Enhanced UI Version
# ==============================

import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="SentiScope — AI Sentiment Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# LOAD MODEL (CACHED FOR PERFORMANCE)
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

# ==============================
# SESSION STATE INITIALIZATION
# ==============================
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []   # Stores past analysis results

# ==============================
# GLOBAL STYLES — Sci-fi lab aesthetic
# ==============================
st.markdown("""
<style>
/* ── Google Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:      #060a12;
    --bg-panel:     #0d1320;
    --border:       rgba(0, 200, 255, 0.15);
    --accent-cyan:  #00c8ff;
    --accent-green: #00ffa3;
    --accent-red:   #ff4060;
    --accent-amber: #ffcc00;
    --text-primary: #e8f0ff;
    --text-muted:   #5a6a8a;
    --font-mono:    'Space Mono', monospace;
    --font-display: 'Syne', sans-serif;
}

/* ── App Background ── */
.stApp {
    background: var(--bg-deep) !important;
    font-family: var(--font-display);
    transition: background 0.6s ease;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1100px; }

/* ── Scrollbar Styling ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 4px; }

/* ── Hero Header ── */
.hero-wrapper {
    text-align: center;
    padding: 3rem 0 1.5rem;
    position: relative;
}
.hero-tag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--accent-cyan);
    background: rgba(0, 200, 255, 0.08);
    border: 1px solid rgba(0, 200, 255, 0.25);
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0.3rem 0;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-muted);
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* ── Divider ── */
.scan-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    margin: 2rem 0;
    opacity: 0.4;
}

/* ── Info Cards Row ── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.info-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.info-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-cyan);
    border-radius: 2px 0 0 2px;
}
.info-card-icon { font-size: 22px; margin-bottom: 6px; }
.info-card-title {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent-cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.info-card-body {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ── Section Labels ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Example Buttons ── */
.stButton > button {
    background: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    padding: 0.5rem 0.8rem !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    text-align: left !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(0, 200, 255, 0.06) !important;
    color: var(--accent-cyan) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 200, 255, 0.12) !important;
}

/* ── Analyze Button (primary) ── */
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button {
    background: linear-gradient(135deg, #004d66, #007a99) !important;
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-size: 13px !important;
    letter-spacing: 2px !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.2rem !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button:hover {
    background: linear-gradient(135deg, #006680, #00a3cc) !important;
    box-shadow: 0 0 30px rgba(0, 200, 255, 0.25) !important;
}

/* ── Text Area ── */
.stTextArea textarea {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.3s ease !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 200, 255, 0.08) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; }
label[data-testid="stWidgetLabel"] { display: none !important; }

/* ── Word/Char counter ── */
.text-counter {
    display: flex;
    justify-content: flex-end;
    gap: 1.5rem;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    margin-top: -0.5rem;
    margin-bottom: 1rem;
    padding-right: 4px;
}

/* ── Result Card ── */
.result-card {
    background: var(--bg-panel);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid var(--border);
    margin-top: 2rem;
    position: relative;
    overflow: hidden;
}
.result-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.result-card.positive::after { background: linear-gradient(90deg, var(--accent-green), #00ffd5); }
.result-card.negative::after { background: linear-gradient(90deg, var(--accent-red), #ff8060); }
.result-card.neutral::after  { background: linear-gradient(90deg, var(--accent-amber), #ffe066); }

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 22px;
    border-radius: 50px;
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 1.5rem;
}
.result-badge.positive {
    background: rgba(0, 255, 163, 0.1);
    border: 1.5px solid var(--accent-green);
    color: var(--accent-green);
}
.result-badge.negative {
    background: rgba(255, 64, 96, 0.1);
    border: 1.5px solid var(--accent-red);
    color: var(--accent-red);
}
.result-badge.neutral {
    background: rgba(255, 204, 0, 0.1);
    border: 1.5px solid var(--accent-amber);
    color: var(--accent-amber);
}

/* ── Confidence Bars ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.conf-label {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    width: 70px;
    flex-shrink: 0;
}
.conf-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.conf-bar-fill.positive { background: linear-gradient(90deg, #00ffa3, #00ffd5); }
.conf-bar-fill.negative { background: linear-gradient(90deg, #ff4060, #ff8060); }
.conf-bar-fill.neutral  { background: linear-gradient(90deg, #ffcc00, #ffe066); }
.conf-pct {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-primary);
    width: 45px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Stats Chips ── */
.stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.stat-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
}
.stat-chip b { color: var(--text-primary); }

/* ── History Panel ── */
.history-item {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 12px;
}
.history-text {
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 70%;
}
.history-badge {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    flex-shrink: 0;
}
.history-badge.positive { background: rgba(0,255,163,0.12); color: var(--accent-green); }
.history-badge.negative { background: rgba(255,64,96,0.12);  color: var(--accent-red); }
.history-badge.neutral  { background: rgba(255,204,0,0.12);  color: var(--accent-amber); }

/* ── Warning / Info ── */
.stAlert { border-radius: 10px !important; font-family: var(--font-mono) !important; font-size: 13px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 2px;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# DYNAMIC BACKGROUND FUNCTION
# ==============================
def set_bg(color_css: str):
    """Injects dynamic background CSS into the app."""
    st.markdown(f"""
    <style>
    .stApp {{ background: {color_css} !important; transition: background 0.6s ease; }}
    </style>
    """, unsafe_allow_html=True)

# Default dark background on load
set_bg("#060a12")


# ==============================
# PREPROCESSING & PREDICTION
# ==============================
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text: str) -> str:
    """Lowercase, remove special characters, strip stopwords."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict(text: str):
    """Return (label_str, probabilities_array | None)."""
    clean = preprocess(text)
    vector = tfidf.transform([clean])
    pred = model.predict(vector)[0]

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if isinstance(pred, (int, float)):
        pred = label_map[int(pred)]

    try:
        probs = model.predict_proba(vector)[0]
    except Exception:
        probs = None

    return pred, probs


# ==============================
# HERO HEADER with custom font
# ==============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Poppins:wght@700&display=swap" rel="stylesheet">

<div class="hero-wrapper" style="width:100%; display:flex; flex-direction:column; align-items:center; text-align:center; margin:0 auto;">
    <div class="hero-tag" style="font-family: 'Poppins', sans-serif;">🔬 NLP · Machine Learning · Real-time</div>
    <div class="hero-title" style="font-family: 'Montserrat', sans-serif; font-size:100px; max-width:600px; margin:0 auto;">
        Senti<span>Scope</span>
    </div>
    <div class="hero-subtitle" style="font-family: 'Poppins', sans-serif;">// An AI Sentiment Analyzer //</div>
    <div class="hero-subtitle" style="font-family: 'Poppins', sans-serif;">Analyze emotions in text using Machine Learning & NLP</div>        
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center; font-family:'Poppins', sans-serif; font-size:16px; color:#cfd8e8; margin-top:10px; max-width:800px; margin-left:auto; margin-right:auto;">
SentiScope is an AI-powered sentiment analyzer that detects emotions in text in real-time. Simply input reviews, feedback, or social media posts, and the app predicts whether the sentiment is Positive, Neutral, or Negative, with confidence scores and interactive visualizations. Keep track of your analyses with the session history and explore quick example texts to get started instantly.
</div>
""", unsafe_allow_html=True)


# ==============================
# INFO CARDS
# ==============================
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <div class="info-card-icon" style="font-size:28px;">🧠</div>
        <div class="info-card-title" style="font-size:18px;" >ML-Powered</div>
        <div class="info-card-body" style="font-size:16px;  color:#ffffff;">TF-IDF vectorization with a trained classifier for accurate tone detection.</div>
    </div>
    <div class="info-card">
        <div class="info-card-icon" style="font-size:28px;">⚡</div>
        <div class="info-card-title" style="font-size:18px;">Real-time</div>
        <div class="info-card-body" style="font-size:16px; color:#ffffff;">Instant predictions with confidence scores across Positive, Neutral & Negative.</div>
    </div>
    <div class="info-card">
        <div class="info-card-icon" style="font-size:28px;">📜</div>
        <div class="info-card-title" style="font-size:18px;">History Log</div>
        <div class="info-card-body" style="font-size:16px; color:#ffffff;">Every analysis is tracked in-session so you can compare results at a glance.</div>
    </div>
</div>
<div class="scan-line"></div>
""", unsafe_allow_html=True)


# ==============================
# EXAMPLE BUTTONS
# ==============================

# ✅ Strong override to REMOVE any green color completely
st.markdown("""
<style>
/* Force all example buttons to neutral style */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* Hover = cyan (NOT green) */
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(0, 200, 255, 0.06) !important;
    color: #00c8ff !important;
    border-color: #00c8ff !important;
}

/* Remove ANY green glow / focus */
div[data-testid="stHorizontalBlock"] .stButton > button:focus,
div[data-testid="stHorizontalBlock"] .stButton > button:active {
    box-shadow: 0 0 15px rgba(0, 200, 255, 0.5) !important;
    border-color: #00c8ff !important;
    color: #00c8ff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label" style="font-size:18px;">⚡ Quick Examples</div>', unsafe_allow_html=True)

examples = {
    "😊 Positive": "This product is absolutely amazing! I loved every bit of it.",
    "😡 Negative": "Worst experience ever. Totally disappointed and frustrated.",
    "😐 Neutral":  "The product is okay, it works as expected.",
    "🤔 Mixed":    "The design is gorgeous and feels premium, but the battery drains way too fast.",
}

cols = st.columns(4)
for col, (label, example_text) in zip(cols, examples.items()):
    if col.button(label):
        st.session_state["_prefill"] = example_text
        st.rerun()

if "_prefill" in st.session_state:
    st.session_state.input_text = st.session_state.pop("_prefill")


# ==============================
# TEXT INPUT AREA
# ==============================
st.markdown('<div class="section-label" style="margin-top:1.5rem; font-size:18px;">✍️ Input Your Text</div>', unsafe_allow_html=True)

# Use a value= default driven by session state; no key= to avoid widget-state conflicts
user_input = st.text_area(
    label="",
    value=st.session_state.get("input_text", ""),
    height=180,
    placeholder="Paste a review, tweet, feedback, or any text here…",
)
# Keep session state in sync so clear works correctly
st.session_state.input_text = user_input

# Live character / word counter
char_count = len(user_input)
word_count = len(user_input.split()) if user_input.strip() else 0
st.markdown(f"""
<div class="text-counter">
    <span>{char_count} chars</span>
    <span>{word_count} words</span>
</div>
""", unsafe_allow_html=True)


# ==============================
# ACTION BUTTONS (Analyze + Clear)
# ==============================

# ✅ Custom CSS ONLY for Analyze button (first column)
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button {
    background: linear-gradient(135deg, #003d2e, #006644) !important;
    border: 1px solid #00ffa3 !important;
    color: #00ffa3 !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}

div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button:hover {
    background: linear-gradient(135deg, #005c3d, #00a36c) !important;
    box-shadow: 0 0 25px rgba(0, 255, 163, 0.35) !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


col_analyze, col_clear = st.columns([6, 1])

with col_analyze:
    analyze = st.button("⟡  ANALYZE SENTIMENT", use_container_width=True)

with col_clear:
    clear = st.button("✕", use_container_width=True)


# ── Clear Handler ──
if clear:
    st.session_state.input_text = ""
    st.session_state.result = None
    set_bg("#060a12")
    st.rerun()


# ── Analyze Handler ──
if analyze:
    if not user_input.strip():
        st.warning("⚠️  Please enter some text before analyzing.")
    else:
        with st.spinner("Scanning sentiment patterns…"):
            prediction, probs = predict(user_input)
            st.session_state.result = (prediction, probs, user_input)

            # Append to session history (keep last 10)
            st.session_state.history.insert(0, {
                "text":  user_input[:80] + ("…" if len(user_input) > 80 else ""),
                "label": prediction,
            })
            st.session_state.history = st.session_state.history[:10]


# ==============================
# RESULT DISPLAY
# ==============================
if st.session_state.result:
    prediction, probs, text = st.session_state.result

    # Gradient map for sentiment
    bg_map = {
        "Positive": "radial-gradient(ellipse at top, #051a10 0%, #060a12 100%)",
        "Negative": "radial-gradient(ellipse at top, #1a0508 0%, #060a12 100%)",
        "Neutral":  "radial-gradient(ellipse at top, #1a1500 0%, #060a12 100%)",
    }

    css_class  = prediction.lower()
    emoji_map  = {"Positive": "😊", "Negative": "😡", "Neutral": "😐"}
    emoji      = emoji_map[prediction]

    # Removed the scan-line
    # st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="font-size:18px;">📊 Analysis Result</div>', unsafe_allow_html=True)

    # Build confidence bars HTML fragment
    conf_html = ""
    if probs is not None:
        conf_html += '<div style="margin-bottom:1.4rem; text-align:center;">'
        conf_html += '<div style="font-family:\'Montserrat\', sans-serif; font-size:14px; color:#ffffff; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">Confidence Breakdown</div>'
        for lbl, idx, bar_cls in [("Negative", 0, "negative"), ("Neutral", 1, "neutral"), ("Positive", 2, "positive")]:
            pct = probs[idx] * 100
            conf_html += (
                f'<div class="conf-row" style="display:flex; justify-content:center; align-items:center; margin-bottom:5px;">'
                f'<span class="conf-label" style="width:80px; text-align:right; margin-right:10px; font-family:\'Montserrat\', sans-serif;">{lbl}</span>'
                f'<div class="conf-bar-bg" style="flex:1;"><div class="conf-bar-fill {bar_cls}" style="width:{pct:.1f}%"></div></div>'
                f'<span class="conf-pct" style="width:50px; text-align:left; margin-left:10px; font-family:\'Montserrat\', sans-serif;">{pct:.1f}%</span>'
                f'</div>'
            )
        conf_html += "</div>"

    # Build text stats chips HTML fragment with Montserrat, bigger, centered
    word_list    = text.split()
    sentences    = len([s for s in re.split(r'[.!?]', text) if s.strip()])
    avg_word_len = round(sum(len(w) for w in word_list) / len(word_list), 1) if word_list else 0
    stats_html = (
        f'<div class="stats-row" style="display:flex; justify-content:center; gap:20px; margin-top:10px;">'
        f'<div class="stat-chip" style="font-size:16px; font-family:\'Montserrat\', sans-serif; color:#ffffff;"><b>{len(text)}</b> charecters</div>'
        f'<div class="stat-chip" style="font-size:16px; font-family:\'Montserrat\', sans-serif; color:#ffffff;"><b>{len(word_list)}</b> words</div>'
        f'<div class="stat-chip" style="font-size:16px; font-family:\'Montserrat\', sans-serif; color:#ffffff;"><b>{sentences}</b> sentences</div>'
        f'<div class="stat-chip" style="font-size:16px; font-family:\'Montserrat\', sans-serif; color:#ffffff;">avg word <b>{avg_word_len}</b> charecters</div>'
        f'</div>'
    )

    # Import Google Font
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    # Render the ENTIRE result card with full background gradient
    st.markdown(
        f'<div class="result-card {css_class}" style="text-align:center; background:{bg_map.get(prediction, "#060a12")}; padding:25px; border-radius:15px; color:#ffffff;">'
        f'<div class="result-badge {css_class}" style="margin:auto; font-size:35px; font-weight:700; font-family:\'Montserrat\', sans-serif; letter-spacing:1px;">'
        f'{emoji}&nbsp;&nbsp;{prediction} Sentiment</div>'
        f'{conf_html}'
        f'{stats_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==============================
# HISTORY LOG
# ==============================
if st.session_state.history:
    st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="font-size:18px;">📜 Session History</div>', unsafe_allow_html=True)

    for item in st.session_state.history:
        css = item["label"].lower()
        emoji = {"Positive": "😊", "Negative": "😡", "Neutral": "😐"}.get(item["label"], "")
        st.markdown(f"""
        <div class="history-item">
            <span class="history-text">{item['text']}</span>
            <span class="history-badge {css}">{emoji} {item['label']}</span>
        </div>
        """, unsafe_allow_html=True)

    # Clear history button
    if st.button("🗑️  Clear History"):
        st.session_state.history = []
        st.rerun()


# ==============================
# FOOTER
# ==============================
st.markdown("""
<div class="scan-line" style="margin-top:3rem;"></div>
<div class="footer" style="
    text-align:center; 
    padding:12px 0; 
    font-size:14px; 
    color:#aaa; 
    background-color:#060a12; 
    border-top:1px solid #333;
">
    ⚡ <strong>SentiScope v2.0</strong> &nbsp;·&nbsp; Powered by ML & NLP & Streamlit &nbsp;·&nbsp; All analyses run locally &nbsp;·&nbsp; Designed for accurate, real-time sentiment detection
</div>
""", unsafe_allow_html=True)