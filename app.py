import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Potato Disease AI",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Root Variables ---- */
:root {
    --green-dark:  #1a3a2a;
    --green-mid:   #2d6a4f;
    --green-light: #52b788;
    --green-pale:  #b7e4c7;
    --cream:       #f8f5ee;
    --text-dark:   #1c1c1c;
    --text-muted:  #6b7280;
    --blight-red:  #c0392b;
    --blight-amber:#e67e22;
    --shadow-soft: 0 4px 24px rgba(0,0,0,0.07);
    --shadow-card: 0 8px 40px rgba(26,58,42,0.10);
    --radius-lg:   20px;
    --radius-md:   12px;
}

/* ---- Global Reset ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--cream) !important;
    color: var(--text-dark) !important;
}
.stApp { background-color: var(--cream) !important; }

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1200px !important;
}

/* ---- Hero Banner ---- */
.hero-banner {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    border-radius: var(--radius-lg);
    padding: 3rem 2.5rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "🌿";
    position: absolute;
    font-size: 180px;
    right: -20px;
    top: -30px;
    opacity: 0.07;
}
.hero-banner::after {
    content: "🥔";
    position: absolute;
    font-size: 140px;
    right: 120px;
    bottom: -30px;
    opacity: 0.07;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--green-pale);
    font-weight: 400;
    margin: 0 0 1.2rem 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* ---- Section Cards ---- */
.section-card {
    background: #ffffff;
    border-radius: var(--radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-card);
    border: 1px solid rgba(82,183,136,0.10);
    margin-bottom: 1.5rem;
    height: 100%;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--green-dark);
    margin: 0 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-divider {
    height: 3px;
    width: 40px;
    background: linear-gradient(90deg, var(--green-mid), var(--green-light));
    border-radius: 2px;
    margin-bottom: 1.2rem;
}

/* ---- Upload Zone ---- */
.stFileUploader > div > div {
    border: 2px dashed var(--green-light) !important;
    border-radius: var(--radius-md) !important;
    background: linear-gradient(135deg, rgba(82,183,136,0.05), rgba(183,228,199,0.08)) !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stFileUploader > div > div:hover {
    border-color: var(--green-mid) !important;
    background: rgba(82,183,136,0.10) !important;
}

/* ---- Predict Button ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--green-light)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(45,106,79,0.3) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(45,106,79,0.4) !important;
}

/* ---- Result Cards ---- */
.result-healthy {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #10b981;
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.result-disease {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.result-label-healthy { color: #065f46; }
.result-label-disease { color: #991b1b; }
.result-caption {
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 0.3rem;
    opacity: 0.8;
}

/* ---- Confidence Bar ---- */
.confidence-wrap {
    background: rgba(82,183,136,0.10);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.confidence-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.confidence-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--green-dark);
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--green-mid), var(--green-light)) !important;
    border-radius: 999px !important;
}

/* ---- Class Probabilities ---- */
.prob-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(82,183,136,0.10);
    font-size: 0.92rem;
    font-weight: 500;
}
.prob-row:last-child { border-bottom: none; }
.prob-name { color: var(--text-dark); }
.prob-pct  { color: var(--green-mid); font-weight: 700; font-size: 1rem; }

/* ---- Info Box ---- */
.info-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
}
.info-empty .icon { font-size: 3rem; margin-bottom: 0.6rem; }
.info-empty p { font-size: 0.95rem; margin: 0; }

/* ---- Disease Info Cards ---- */
.disease-card {
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.disease-card-eb  { background: #fff7ed; border-left: 4px solid var(--blight-amber); }
.disease-card-lb  { background: #fef2f2; border-left: 4px solid var(--blight-red); }
.disease-card-h   { background: #f0fdf4; border-left: 4px solid #22c55e; }
.disease-card h4  { margin: 0 0 0.3rem 0; font-size: 0.95rem; font-weight: 700; }
.disease-card p   { margin: 0; font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; }

/* ---- About Section ---- */
.about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
}
.stat-tile {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid));
    border-radius: var(--radius-md);
    padding: 1.2rem;
    text-align: center;
    color: white;
}
.stat-tile .stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 900;
    display: block;
}
.stat-tile .stat-lbl {
    font-size: 0.78rem;
    opacity: 0.8;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.tech-pill {
    display: inline-block;
    background: rgba(45,106,79,0.10);
    color: var(--green-dark);
    border: 1px solid rgba(45,106,79,0.20);
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem;
}

.disease-card-eb, .disease-card-lb, .disease-card-h {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    border: 2px solid #000000;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    color: black;
    margin-top: 10px;
}       

/* ---- Footer ---- */
.footer-strip {
    background: var(--green-dark);
    border-radius: var(--radius-lg);
    padding: 1.5rem 2rem;
    text-align: center;
    color: rgba(255,255,255,0.7);
    font-size: 0.88rem;
    margin-top: 2rem;
}
.footer-strip strong { color: var(--green-pale); }

/* ---- Animations ---- */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-banner, .section-card { animation: fadeSlideUp 0.6s ease both; }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_disease_model.keras")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Class order MUST match what tf.keras.preprocessing.image_dataset_from_directory
# returned during training. That function sorts folder names alphabetically:
#   Potato___Early_blight  → index 0
#   Potato___Late_blight   → index 1
#   Potato___healthy       → index 2
#
# The model also contains a Rescaling(1/255) layer, so we must NOT divide
# pixel values by 255 again during preprocessing — only resize and batch.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

DISEASE_INFO = {
    "Early Blight": {
        "desc": "Caused by Alternaria solani fungus. Appears as dark brown spots with concentric rings forming a target-board pattern on older leaves.",
        "severity": "Moderate",
        "action": "Apply copper-based fungicide, remove affected leaves, ensure proper crop rotation.",
        "card_class": "disease-card-eb",
        "emoji": "🟠"
    },
    "Late Blight": {
        "desc": "Caused by Phytophthora infestans. Rapid dark, water-soaked lesions on leaves that quickly spread and destroy the crop.",
        "severity": "Severe",
        "action": "Apply systemic fungicide immediately, remove infected plants, avoid overhead irrigation.",
        "card_class": "disease-card-lb",
        "emoji": "🔴"
    },
    "Healthy": {
        "desc": "No signs of disease detected. The leaf appears vibrant green with no lesions, spots, or abnormal discoloration.",
        "severity": "None",
        "action": "Continue regular monitoring. Maintain proper irrigation and balanced fertilization.",
        "card_class": "disease-card-h",
        "emoji": "🟢"
    }
}

# =========================
# Preprocessing
# NOTE: The trained model already includes Rescaling(1./255) as the
# second layer, so we pass raw uint8 pixel values (0–255) here.
# Only resize and add the batch dimension.
# =========================
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.resize((256, 256))
    img_array = np.array(img)          # shape (256, 256, 3), dtype uint8
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 256, 256, 3)
    return img_array

# =========================
# Hero Banner
# =========================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🥔 Potato Leaf Disease<br>Detection System</div>
    <div class="hero-sub">AI-powered crop health analysis using Convolutional Neural Networks</div>
    <span class="hero-badge">🎓 Developed by Pranav V P &nbsp;·&nbsp; Internship Project 2026</span>
</div>
""", unsafe_allow_html=True)

# =========================
# Main Two-Column Layout
# =========================
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload a potato leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Uploaded leaf image")
        st.markdown(f"""
        <p style="font-size:0.82rem; color:var(--text-muted); margin-top:0.5rem; text-align:center;">
        📁 {uploaded_file.name} &nbsp;|&nbsp; {uploaded_file.size // 1024} KB
        </p>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Diagnosis Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Analysing leaf tissue…"):
            time.sleep(0.4)  # brief UX pause for realism
            processed = preprocess_image(image)
            predictions = model.predict(processed, verbose=0)

        pred_idx   = int(np.argmax(predictions[0]))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(np.max(predictions[0]))
        info       = DISEASE_INFO[pred_class]

        # --- Result badge ---
        if pred_class == "Healthy":
            st.markdown(f"""
            <div class="result-healthy">
                <div class="result-label result-label-healthy">✅ {pred_class}</div>
                <div class="result-caption" style="color:#065f46;">No disease detected — leaf is healthy</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            severity_color = "#b45309" if pred_class == "Early Blight" else "#991b1b"
            st.markdown(f"""
            <div class="result-disease">
                <div class="result-label result-label-disease">⚠️ {pred_class}</div>
                <div class="result-caption" style="color:{severity_color};">Severity: {info['severity']} — Immediate attention recommended</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Confidence ---
        st.markdown(f"""
        <div class="confidence-wrap">
            <div class="confidence-label">Model Confidence</div>
            <div class="confidence-value">{confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence)

        # --- All class probabilities ---
        st.markdown("<br>**Probability Breakdown**", unsafe_allow_html=True)
        prob_html = ""
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
            bar_width = int(prob * 100)
            highlight = "font-weight:700; color:var(--green-dark);" if i == pred_idx else ""
            prob_html += f"""
            <div class="prob-row">
                <span class="prob-name" style="{highlight}">{DISEASE_INFO[cls]['emoji']} {cls}</span>
                <span class="prob-pct">{prob*100:.1f}%</span>
            </div>
            """
        st.markdown(prob_html, unsafe_allow_html=True)

        # --- Disease details ---
        st.markdown("<br>**Recommended Action**", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="disease-card {info['card_class']}">
            <p>{info['desc']}</p>
            <p style="margin-top:0.5rem; font-weight:600; color:var(--text-dark);">💊 {info['action']}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-empty">
            <div class="icon">🌿</div>
            <p>Upload a potato leaf image on the left<br>to receive an instant AI diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Disease Reference Cards
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Disease Reference Guide</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

ref_cols = st.columns(3)
diseases = [("Early Blight", "disease-card-eb"), ("Late Blight", "disease-card-lb"), ("Healthy", "disease-card-h")]
for col, (name, card_class) in zip(ref_cols, diseases):
    info = DISEASE_INFO[name]
    with col:
        st.markdown(f"""
        <div class="disease-card {card_class}" style="height:100%;">
            <h4>{info['emoji']} {name}</h4>
            <p><b>Severity:</b> {info['severity']}</p>
            <p style="margin-top:0.4rem;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# About the Project
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📘 About the Project</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<p style="font-size:1rem; line-height:1.75; color:var(--text-muted);">
This project develops an <strong>automated AI system</strong> to detect potato leaf diseases using deep learning.
Potato crops are critical to global food security, yet highly vulnerable to fungal pathogens such as
<em>Alternaria solani</em> (Early Blight) and <em>Phytophthora infestans</em> (Late Blight).
Traditional manual inspection is slow, costly, and inaccessible to smallholder farmers.
This system provides instant, accurate, and accessible diagnosis directly from a smartphone or browser.
</p>
""", unsafe_allow_html=True)

stat_cols = st.columns(4)
stats = [
    ("3", "Disease Classes"),
    ("256×256", "Input Resolution"),
    ("CNN", "Architecture"),
    ("PlantVillage", "Dataset"),
]
for col, (val, lbl) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
        <div class="stat-tile">
            <span class="stat-val">{val}</span>
            <span class="stat-lbl">{lbl}</span>
        </div>
        """, unsafe_allow_html=True)

about_l, about_r = st.columns(2)
with about_l:
    st.markdown("""
    <br>
    <h4 style="font-family:'Playfair Display',serif; color:var(--green-dark);">🧠 Model Architecture</h4>
    <p style="color:var(--text-muted); font-size:0.92rem; line-height:1.7;">
    A custom <strong>Convolutional Neural Network (CNN)</strong> was built with TensorFlow/Keras, comprising:
    <ul style="color:var(--text-muted);">
        <li>Data augmentation (random flip, random rotation)</li>
        <li>Built-in rescaling / normalisation layer</li>
        <li>3× Conv2D + MaxPooling2D feature extraction blocks</li>
        <li>Flatten → Dense(64, ReLU) → Dense(3, Softmax) classifier head</li>
        <li>Trained with Adam optimiser and Sparse Categorical Cross-Entropy loss</li>
        <li>15 epochs on 80/10/10 train/val/test splits</li>
    </ul>
    </p>
    """, unsafe_allow_html=True)

with about_r:
    st.markdown("""
    <br>
    <h4 style="font-family:'Playfair Display',serif; color:var(--green-dark);">⚙️ Inference Pipeline</h4>
    <p style="color:var(--text-muted); font-size:0.92rem; line-height:1.7;">
    <ol style="color:var(--text-muted);">
        <li><strong>Upload</strong> – user provides a JPG/PNG leaf image</li>
        <li><strong>Resize</strong> – image is resized to 256×256 pixels</li>
        <li><strong>Batch</strong> – a batch dimension is added for model input</li>
        <li><strong>Inference</strong> – model internally rescales and predicts class probabilities</li>
        <li><strong>Output</strong> – top class, confidence %, and actionable advice are displayed</li>
    </ol>
    </p>
    <h4 style="font-family:'Playfair Display',serif; color:var(--green-dark); margin-top:1rem;">🌱 Impact</h4>
    <p style="color:var(--text-muted); font-size:0.92rem; line-height:1.7;">
    Enables early intervention, reduces crop loss, and democratises precision agriculture diagnostics
    for farmers with limited access to agronomists.
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
<br>
<h4 style="font-family:'Playfair Display',serif; color:var(--green-dark);">🧪 Tech Stack</h4>
""", unsafe_allow_html=True)
for tech in ["TensorFlow 2.x", "Keras", "Streamlit", "NumPy", "Pillow", "scikit-learn", "PlantVillage Dataset", "KaggleHub"]:
    st.markdown(f'<span class="tech-pill">{tech}</span>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer-strip">
    <strong>🥔 Potato Leaf Disease Detection System</strong><br>
    Developed by <strong>Pranav V P</strong> &nbsp;·&nbsp; Internship Project 2026 &nbsp;·&nbsp;
    Powered by TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)