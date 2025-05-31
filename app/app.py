import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
import warnings
import torch.nn.functional as F
# Add root to PYTHONPATH
import sys
from pathlib import Path

# Add root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Avoid OMP error from PyTorch/OpenCV
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Suppress FutureWarning from Matplotlib
warnings.filterwarnings("ignore", category=UserWarning)

# Import custom modules
from models.resnet_model import MalariaResNet50 
from gradcam.gradcam import visualize_gradcam


# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="🧬 Malaria Cell Classifier", layout="wide")
st.title("🧬 Malaria Cell Classifier with Grad-CAM")
st.write("Upload a blood smear image and the model will classify it as infected or uninfected, and highlight key regions using Grad-CAM.")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    # Ensure model class doesn't wrap backbone
    model = MalariaResNet50(num_classes=2)
    model.load_state_dict(torch.load("models/malaria_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    temp_image_path = f"temp_{uploaded_file.name}"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display original image (resize if needed)
    image = Image.open(uploaded_file).convert("RGB")
    max_size = (400, 400)  # Max width and height
    image.thumbnail(max_size)
    st.image(image, caption="Uploaded Image", use_container_width=False)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            # Run prediction
            pred_label, confidence = model.predict(temp_image_path, device='cpu', show_image=False)
            st.success(f"✅ Prediction: **{pred_label}** | Confidence: **{confidence:.2%}**")

            # Show Grad-CAM
            st.subheader("🔍 Grad-CAM Visualization")
            with st.expander("ℹ️ What is Grad-CAM?"):
                st.markdown("""
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** is an interpretability method that shows which parts of an image are most important for a CNN's prediction.
            
                How it works:
                1. Gradients flow from the output neuron back to the last convolutional layer.
                2. These gradients are global average pooled to get importance weights.
                3. A weighted combination creates a coarse heatmap.
                4. Final heatmap is overlaid on the original image.
            
                🔬 In this app:
                - Helps understand *why* the model thinks a blood smear cell is infected
                - Makes predictions more transparent and reliable
                """)
            visualize_gradcam(model, temp_image_path)