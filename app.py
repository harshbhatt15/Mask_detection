import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MaskScan AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #060910;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: '';
    position: fixed;
    top: -200px; right: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
    position: relative;
    z-index: 1;
}

/* ── Hero ── */
.hero { text-align: center; padding: 3rem 0 2rem; }

.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    font-size: 11px; font-weight: 500;
    letter-spacing: 3px; text-transform: uppercase;
    padding: 6px 18px; border-radius: 50px; margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(42px, 6vw, 72px);
    font-weight: 800; line-height: 1.05;
    letter-spacing: -2px; color: #f8fafc; margin-bottom: 0.5rem;
}

.hero-title span {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub { font-size: 16px; color: #64748b; font-weight: 300; margin-top: 1rem; }

/* ── Stats ── */
.stats-row {
    display: flex; justify-content: center;
    gap: 2rem; margin: 2.5rem 0; flex-wrap: wrap;
}

.stat-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 24px;
    text-align: center; min-width: 120px;
}

.stat-chip .num {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 700; color: #38bdf8;
}

.stat-chip .lbl {
    font-size: 11px; color: #475569;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 2px;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent);
    margin: 2rem 0;
}

/* ── Upload ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 13px; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #475569; margin-bottom: 1rem; display: block;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(56,189,248,0.2) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
}

/* ── Panel ── */
.panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 28px;
    position: relative; overflow: hidden;
}

.panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
}

.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px; font-weight: 600;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #334155; margin-bottom: 1.2rem;
}

/* ── Result ── */
.result-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 280px; gap: 1.5rem;
}

.result-icon {
    width: 80px; height: 80px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 36px;
}

.result-icon.safe   { background: rgba(34,197,94,0.1);  border: 2px solid rgba(34,197,94,0.3);  box-shadow: 0 0 40px rgba(34,197,94,0.15); }
.result-icon.danger { background: rgba(239,68,68,0.1);  border: 2px solid rgba(239,68,68,0.3);  box-shadow: 0 0 40px rgba(239,68,68,0.15); }

.result-label { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; text-align: center; }
.result-label.safe   { color: #22c55e; }
.result-label.danger { color: #ef4444; }

.verdict {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 20px; border-radius: 99px;
    font-size: 12px; font-weight: 500;
    letter-spacing: 1px; text-transform: uppercase;
}

.verdict.safe   { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2);  color: #22c55e; }
.verdict.danger { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);  color: #ef4444; }

.conf-wrap { width: 100%; max-width: 260px; }

.conf-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.conf-title  { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #334155; }
.conf-value  { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; color: #38bdf8; }

.conf-bar-bg { height: 6px; background: rgba(255,255,255,0.05); border-radius: 99px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 99px; }
.conf-bar-fill.safe   { background: linear-gradient(90deg, #22c55e, #86efac); }
.conf-bar-fill.danger { background: linear-gradient(90deg, #ef4444, #fca5a5); }

/* ── POPUP OVERLAY ── */
.popup-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(8px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

.popup-card {
    background: #0f172a;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 24px;
    padding: 2.5rem;
    max-width: 420px;
    width: 90%;
    text-align: center;
    box-shadow: 0 0 60px rgba(239,68,68,0.15), 0 25px 50px rgba(0,0,0,0.5);
    animation: slideUp 0.3s ease;
    position: relative;
}

@keyframes slideUp {
    from { transform: translateY(30px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}

.popup-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(239,68,68,0.5), transparent);
    border-radius: 24px 24px 0 0;
}

.popup-icon {
    width: 70px; height: 70px;
    border-radius: 50%;
    background: rgba(239,68,68,0.1);
    border: 2px solid rgba(239,68,68,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; margin: 0 auto 1.5rem;
    box-shadow: 0 0 30px rgba(239,68,68,0.2);
}

.popup-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 800;
    color: #f8fafc; letter-spacing: -0.5px;
    margin-bottom: 0.75rem;
}

.popup-msg {
    font-size: 14px; color: #64748b;
    line-height: 1.7; margin-bottom: 1.5rem;
}

.popup-rules {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem 1.25rem;
    text-align: left; margin-bottom: 1.75rem;
}

.popup-rule {
    font-size: 12px; color: #475569;
    padding: 5px 0; display: flex; align-items: center; gap: 8px;
}

.popup-rule span { color: #38bdf8; font-size: 14px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080c12 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem !important;
}

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 20px; font-weight: 800;
    color: #f8fafc; letter-spacing: -0.5px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.sidebar-logo span { color: #38bdf8; }
.sidebar-section { margin-bottom: 2rem; }

.sidebar-section-title {
    font-size: 10px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #334155;
    margin-bottom: 1rem; font-weight: 600;
}

.info-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.info-row .key   { font-size: 12px; color: #475569; }
.info-row .value { font-size: 12px; color: #94a3b8; font-weight: 500; }

.status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 8px #22c55e; margin-right: 6px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

.step { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 1.2rem; }

.step-num {
    width: 28px; height: 28px; border-radius: 8px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif; font-size: 12px;
    font-weight: 700; color: #38bdf8; flex-shrink: 0;
}

.step-text { font-size: 12px; color: #475569; line-height: 1.6; padding-top: 4px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ------------------ MODEL ------------------
class MaskCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 30 * 30, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.5)
        self.out   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.out(x)

@st.cache_resource
def load_model():
    model = MaskCNN(num_classes=2)
    model.load_state_dict(torch.load("mask_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ------------------ VALIDATION FUNCTION ------------------
def validate_image(uploaded_file):
    """
    Returns (is_valid, error_reason)
    Checks: file size, readable as image, min dimensions, not solid color
    """
    # 1. File size check (max 5MB)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if len(file_bytes) > 5 * 1024 * 1024:
        return False, "file_too_large"

    # 2. Readable as image
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return False, "not_an_image"

    # 3. Minimum size check (at least 30x30)
    w, h = img.size
    if w < 30 or h < 30:
        return False, "too_small"

    # 4. Blank/solid color check
    arr = np.array(img)
    if arr.std() < 5:
        return False, "blank_image"

    return True, None

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Mask<span>Scan</span> AI</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="sidebar-section">
        <div class="sidebar-section-title">System Status</div>
        <div class="info-row"><span class="key"><span class="status-dot"></span>Model</span><span class="value">Online</span></div>
        <div class="info-row"><span class="key">Framework</span><span class="value">PyTorch</span></div>
        <div class="info-row"><span class="key">Architecture</span><span class="value">CNN</span></div>
        <div class="info-row"><span class="key">Input Size</span><span class="value">128 × 128</span></div>
        <div class="info-row"><span class="key">Classes</span><span class="value">Mask / No Mask</span></div>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">Training Info</div>
        <div class="info-row"><span class="key">Dataset</span><span class="value">16,584 images</span></div>
        <div class="info-row"><span class="key">Epochs</span><span class="value">10</span></div>
        <div class="info-row"><span class="key">Optimizer</span><span class="value">Adam</span></div>
        <div class="info-row"><span class="key">Batch Size</span><span class="value">32</span></div>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">How It Works</div>
        <div class="step"><div class="step-num">1</div><div class="step-text">Upload a face image in JPG, PNG or JPEG format</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text">Image resized to 128×128 and normalized</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text">CNN runs inference and returns prediction</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text">Result shown with confidence score instantly</div></div>
    </div>
    ''', unsafe_allow_html=True)

# ------------------ HERO ------------------
st.markdown('''
<div class="hero">
    <div class="hero-badge">🛡️ &nbsp; AI-Powered Detection</div>
    <div class="hero-title">Face Mask<br><span>Detection</span></div>
    <div class="hero-sub">Upload a face image — get an instant AI-powered mask detection result</div>
</div>
<div class="stats-row">
    <div class="stat-chip"><div class="num">16,584</div><div class="lbl">Images Trained</div></div>
    <div class="stat-chip"><div class="num">CNN</div><div class="lbl">Architecture</div></div>
    <div class="stat-chip"><div class="num">2</div><div class="lbl">Classes</div></div>
    <div class="stat-chip"><div class="num">128px</div><div class="lbl">Input Size</div></div>
</div>
<div class="divider"></div>
''', unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
st.markdown('<span class="upload-label">Upload Image</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# ------------------ POPUP STATE ------------------
if "show_popup" not in st.session_state:
    st.session_state.show_popup = False
if "popup_reason" not in st.session_state:
    st.session_state.popup_reason = None

# ------------------ POPUP MESSAGES ------------------
POPUP_CONTENT = {
    "file_too_large": {
        "icon": "📦",
        "title": "File Too Large",
        "msg": "The file you uploaded exceeds the 5MB limit. Please upload a smaller image.",
        "rules": [
            ("✓", "Maximum file size: 5MB"),
            ("✓", "Use JPG or PNG format for smaller sizes"),
            ("✓", "Compress the image before uploading"),
        ]
    },
    "not_an_image": {
        "icon": "🚫",
        "title": "Invalid File",
        "msg": "The file you uploaded could not be read as an image. Please upload a valid image file.",
        "rules": [
            ("✓", "Supported formats: JPG, PNG, JPEG"),
            ("✓", "Do not upload PDF, Word, or video files"),
            ("✓", "Make sure the file is not corrupted"),
        ]
    },
    "too_small": {
        "icon": "🔍",
        "title": "Image Too Small",
        "msg": "The image resolution is too low for accurate detection. Please upload a clearer image.",
        "rules": [
            ("✓", "Minimum size: 30 × 30 pixels"),
            ("✓", "Higher resolution gives better results"),
            ("✓", "Avoid heavily cropped or thumbnail images"),
        ]
    },
    "blank_image": {
        "icon": "⬜",
        "title": "Blank or Invalid Image",
        "msg": "The uploaded image appears to be blank or a solid color. Please upload a real face photo.",
        "rules": [
            ("✓", "Upload a real face photograph"),
            ("✓", "Make sure the image is not all white or black"),
            ("✓", "Ensure proper lighting in the photo"),
        ]
    },
}

# ------------------ VALIDATION & POPUP ------------------
if uploaded_file is not None:
    is_valid, reason = validate_image(uploaded_file)

    if not is_valid:
        st.session_state.show_popup = True
        st.session_state.popup_reason = reason

# ------------------ SHOW POPUP ------------------
if st.session_state.show_popup and st.session_state.popup_reason:
    content = POPUP_CONTENT[st.session_state.popup_reason]

    st.markdown(f'''
    <div class="popup-overlay">
        <div class="popup-card">
            <div class="popup-icon">{content["icon"]}</div>
            <div class="popup-title">{content["title"]}</div>
            <div class="popup-msg">{content["msg"]}</div>
            <div class="popup-rules">
                {"".join(f'<div class="popup-rule"><span>{r[0]}</span>{r[1]}</div>' for r in content["rules"])}
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("✕  Dismiss & Try Again", use_container_width=True):
            st.session_state.show_popup = False
            st.session_state.popup_reason = None
            st.rerun()

# ------------------ RESULTS ------------------
elif uploaded_file is not None:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="panel"><div class="panel-title">Input Image</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title">Analysis Result</div>', unsafe_allow_html=True)

        img        = image.resize((128, 128))
        img_array  = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

        with st.spinner("Running inference..."):
            with torch.no_grad():
                outputs    = model(img_tensor)
                probs      = torch.softmax(outputs, dim=1)[0]
                class_idx  = torch.argmax(probs).item()
                confidence = probs[class_idx].item()

        css_class = "safe"   if class_idx == 1 else "danger"
        icon      = "😷"     if class_idx == 1 else "🚫"
        label     = "Mask Detected"    if class_idx == 1 else "No Mask Detected"
        verdict   = "Protected"        if class_idx == 1 else "Unprotected"
        conf_pct  = f"{confidence * 100:.1f}%"

        st.markdown(f'''
        <div class="result-wrap">
            <div class="result-icon {css_class}">{icon}</div>
            <div class="result-label {css_class}">{label}</div>
            <div class="verdict {css_class}">● &nbsp; {verdict}</div>
            <div class="conf-wrap">
                <div class="conf-header">
                    <span class="conf-title">Confidence</span>
                    <span class="conf-value">{conf_pct}</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill {css_class}" style="width:{conf_pct}"></div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('''
    <div style="text-align:center; padding:4rem 2rem;
        border: 2px dashed rgba(56,189,248,0.1);
        border-radius: 20px; margin-top: 1rem;">
        <div style="font-size:48px; margin-bottom:1rem;">🛡️</div>
        <div style="font-family:Syne,sans-serif; font-size:18px;
            font-weight:700; color:#1e293b; margin-bottom:0.5rem;">
            No Image Uploaded
        </div>
        <div style="font-size:13px; color:#334155;">
            Upload a face image above to get started
        </div>
    </div>
    ''', unsafe_allow_html=True)
