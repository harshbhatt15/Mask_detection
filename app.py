import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Mask Detection", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: #0f172a;
}

/* Main container */
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 2rem;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #f8fafc;
}

/* Card */
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
    margin-top: 20px;
}

/* Result text */
.success {
    color: #22c55e;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}

.error {
    color: #ef4444;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------ MODEL DEFINITION ------------------
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

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = MaskCNN(num_classes=2)
    model.load_state_dict(torch.load("mask_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ------------------ HEADER ------------------
st.markdown('<p class="main-title">😷 Face Mask Detection</p>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 About")
st.sidebar.write("This app detects whether a person is wearing a mask using a CNN model.")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Model: mask_model.pth")

# ------------------ FILE UPLOAD ------------------
st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # ------------------ IMAGE DISPLAY ------------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ PREDICTION ------------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Preprocess: resize → numpy float32 → tensor → CHW → batch
        img = image.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0          # (128,128,3)
        img_tensor = torch.tensor(img_array).permute(2, 0, 1)        # (3,128,128)
        img_tensor = img_tensor.unsqueeze(0)                          # (1,3,128,128)

        with st.spinner("Analyzing image..."):
            with torch.no_grad():
                outputs     = model(img_tensor)                       # raw logits
                probs       = torch.softmax(outputs, dim=1)[0]        # probabilities
                class_idx   = torch.argmax(probs).item()
                confidence  = probs[class_idx].item()

        st.markdown("### 🎯 Result")

        if class_idx == 1:
            st.markdown(f'<p class="success">✅ Mask Detected</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="error">❌ No Mask Detected</p>', unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align:center; color:#cbd5e1;'>Confidence: {confidence*100:.2f}%</p>",
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)