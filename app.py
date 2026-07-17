import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Image Captioning + YOLO Detection", layout="wide")

st.title("📷 Image Captioning using BLIP + YOLOv8")
st.write("Upload an image to generate object detection and an automatic caption.")

# Load YOLO model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

# Load BLIP model
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return processor, model, device


yolo_model = load_yolo()
processor, blip_model, device = load_blip()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Results"):

        # -----------------------
        # YOLO Detection
        # -----------------------
        results = yolo_model.predict(
            source=np.array(image),
            conf=0.6
        )

        plotted = results[0].plot()

        st.subheader("Detected Objects")
        st.image(plotted, use_container_width=True)

        objects = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            objects.append(yolo_model.names[cls])

        if len(objects):
            st.success("Objects Detected:")
            st.write(", ".join(set(objects)))
        else:
            st.warning("No objects detected.")

        # -----------------------
        # BLIP Caption
        # -----------------------
        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        output = blip_model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=3,
            repetition_penalty=1.4,
            length_penalty=1.2,
            early_stopping=True
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        st.subheader("Generated Caption")
        st.success(caption)
