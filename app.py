import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import gdown
import os

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = Path("model/resnet50_melanoma_best.pth")
DRIVE_URL = "https://drive.google.com/uc?id=1LbxdBGKK4B2gO-m_F8EsNmVvuiHSWqfM"  # <-- Converted to direct link
IMG_SIZE = 224

# -------------------------------
# Ensure Model Exists
# -------------------------------
def download_model():
    os.makedirs("model", exist_ok=True)
    if not MODEL_PATH.exists():
        with st.spinner("📦 Downloading model from Google Drive (first time only)..."):
            gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=False)
        st.success("✅ Model downloaded successfully!")

# -------------------------------
# Model Definition
# -------------------------------
@st.cache_resource
def load_model():
    download_model()
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    try:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# -------------------------------
# Prediction
# -------------------------------
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        prob = torch.sigmoid(outputs).item()
        label = 1 if prob >= 0.5 else 0
        return label, prob

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Melanoma Detection", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef1f5 100%);
        font-family: 'Inter', sans-serif;
        color: #2d3436;
    }
    h1.main-title {
        text-align: center;
        font-size: 2.4rem;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0.4em;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.5em;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    [data-testid="stFileUploader"] section div div button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"] section div div button:hover {
        background-color: #1f2937 !important;
        border-color: #000000 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 class='main-title'>🩺 Melanoma Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a dermoscopic image to predict if it’s <b>Melanoma</b> or <b>Benign</b>.</p>", unsafe_allow_html=True)

# -------------------------------
# Upload Section
# -------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Inference
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image... please wait"):
        model = load_model()
        if model is not None:
            image_tensor = preprocess_image(image)
            label, prob = predict(model, image_tensor)

            st.subheader("🔍 Prediction Result:")
            if label == 1:
                st.markdown("### 🩸 **Melanoma Detected**")
                st.write(f"**Confidence:** {prob * 100:.2f}%")
                st.progress(float(prob))
            else:
                st.markdown("### 🌿 **Benign (No Melanoma)**")
                st.write(f"**Confidence:** {(1 - prob) * 100:.2f}%")
                st.progress(float(1 - prob))
else:
    st.info("Please upload an image to start the analysis.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class='footer'>
    Made with <b>Streamlit</b> + <b>PyTorch</b><br>
    <span style='font-size:0.8rem;'>AI-based skin lesion classifier (for educational use)</span>
</div>
""", unsafe_allow_html=True)
