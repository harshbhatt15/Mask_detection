# 😷 Face Mask Detection

A deep learning web application that detects whether a person is wearing a face mask or not, built with **PyTorch** and deployed using **Streamlit**.

---

## 🌐 Live Demo

👉 [https://maskdetection-vbbjp8fwpmhcpcq2nzdazm.streamlit.app](https://maskdetection-vbbjp8fwpmhcpcq2nzdazm.streamlit.app)

---

## 📸 Preview

Upload any face image and instantly get:
- ✅ **Mask Detected**
- ❌ **No Mask Detected**

with a confidence score.

---

## 📂 Dataset

| Class | Images |
|-------|--------|
| With Mask | 3,725 |
| Without Mask | 3,828 |
| **Total** | **7,553** |

- Images resized to **128×128 pixels**
- Converted to RGB format
- 80/20 Train/Test split

---

## 🧠 Model Architecture

```
Input (3, 128, 128)
    ↓
Conv2D(32, 3×3) + ReLU + MaxPool2D
    ↓
Conv2D(64, 3×3) + ReLU + MaxPool2D
    ↓
Flatten → Dense(128) + ReLU + Dropout(0.5)
    ↓
Dense(64) + ReLU + Dropout(0.5)
    ↓
Output(2) → Softmax
```

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 32 |
| Input Size | 128×128×3 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model Training | PyTorch |
| Data Processing | NumPy, Pillow, scikit-learn |
| Notebook | Jupyter / VS Code |
| Containerization | Docker |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
Mask-detection/
├── app.py                        # Streamlit web app
├── mask_detection_pytorch.ipynb  # Training notebook
├── mask_model.pth                # Saved PyTorch model
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose config
└── wheels/                       # Local PyTorch wheels
```

---

## 🚀 Run Locally

### Option 1 — Streamlit App (recommended)

```bash
# Clone the repo
git clone https://github.com/harshbhatt15/Mask_detection.git
cd Mask_detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

### Option 2 — Jupyter Notebook via Docker

```bash
# Build and start
docker-compose up --build

# Open in browser
http://localhost:8888
```

---

## 📦 Requirements

```txt
torch
torchvision
streamlit
Pillow
numpy
scikit-learn
matplotlib
```

---

## 🔄 Workflow

```
Raw Images (7,553)
    ↓
Resize (128×128) + Normalize (/255)
    ↓
Train/Test Split (80/20)
    ↓
CNN Training — 10 Epochs (PyTorch)
    ↓
Save mask_model.pth
    ↓
Streamlit Web App
    ↓
Deploy → Streamlit Cloud 🌍
```

---

## 🐳 Docker Setup

The Docker container runs a **Jupyter notebook server** for interactive training:

- Base image: `python:3.10-slim`
- Port: `8888`
- Data mounted as volume (not baked into image)
- CPU-only PyTorch for lightweight container

```bash
docker-compose up    # start
docker-compose down  # stop
```

---

## 🔮 Future Improvements

- 📹 Real-time webcam detection using OpenCV
- 🏗️ Transfer learning with ResNet / MobileNet for higher accuracy
- 🎭 Multi-class detection (mask types — surgical, N95, cloth)
- ⚡ REST API using FastAPI for system integration
- 📱 Mobile-friendly UI

---

## 👨‍💻 Author

**Harsh Bhatt**
- GitHub: [@harshbhatt15](https://github.com/harshbhatt15)

---
