import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------------------------------------------------------------
# 1. Page Configuration & Title
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 & BLIP Image AI Dashboard",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ YOLOv8 & BLIP Image AI Dashboard")
st.markdown(
    "Upload an image to perform raw preprocessing, real-time object detection using **YOLOv8**, "
    "and natural captioning with the **BLIP Transformer model**."
)
st.write("---")

# -----------------------------------------------------------------------------
# 2. Sidebar Settings & Caching Models
# -----------------------------------------------------------------------------
st.sidebar.header("🔧 Model Settings")

# Cache models to keep user interactions snappy and avoid reloading every rerun
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

# Load both models
with st.spinner("Initializing AI Models..."):
    yolo_model = load_yolo_model()
    blip_processor, blip_model, device = load_blip_model()

# Sliders for YOLO predictions
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.60, 0.05)
iou_threshold = st.sidebar.slider("YOLO IOU Threshold", 0.0, 1.0, 0.60, 0.05)

# -----------------------------------------------------------------------------
# 3. File Upload Interface
# -----------------------------------------------------------------------------
uploaded_file = st.file_sidebar = st.file_uploader(
    "Choose an image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the image
    raw_image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(raw_image)
    
    # -------------------------------------------------------------------------
    # Tab 1: Image Preprocessing (From your manual notebook pipeline)
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🖼️ Preprocessing Details", "🎯 Object Detection", "📝 AI Captioning"])
    
    with tab1:
        st.subheader("Manual Preprocessing Visualizer")
        st.write(f"**Original Image Dimensions:** {img_np.shape[1]}x{img_np.shape[0]} ({img_np.shape[2]} channels)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(raw_image, caption="Original Uploaded Image", use_container_width=True)
            
        with col2:
            # Recreate your notebook pipeline steps:
            # 1. Resize to 224x224
            img_resized = cv2.resize(img_np, (224, 224))
            # 2. Scale
            img_scaled = img_resized.astype(np.float32) / 255.0
            # 3. Normalize (using standard ImageNet statistics)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # Fixed order for RGB
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_scaled - mean) / std
            # 4. Create Batch Tensor
            final_tensor = np.expand_dims(img_normalized, axis=0)
            
            st.image(img_resized, caption="Step 1: Resized Canvas (224 x 224)", use_container_width=True)
            
        st.info(f"**Generated Preprocessed Input Tensor Shape:** {final_tensor.shape}")
        
    # -------------------------------------------------------------------------
    # Tab 2: YOLOv8 Object Detection
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("YOLOv8 Detection Outputs")
        
        # Run inference on BGR formatted frame (OpenCV style)
        cv2_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = yolo_model.predict(
            source=cv2_img, 
            conf=conf_threshold, 
            iou=iou_threshold
        )
        
        # Render the predictions onto the canvas
        for r in results:
            im_array = r.plot()  # Draws bounding boxes and labels
            # Convert BGR back to RGB for streamlit visualization
            annotated_image = Image.fromarray(im_array[..., ::-1])
            
            st.image(annotated_image, caption="Inference Output With Bounding Boxes", use_container_width=True)
            
            # Print Detected Object Data in structured markdown
            st.write("### Detected Objects Details:")
            boxes = r.boxes
            if len(boxes) == 0:
                st.write("No objects detected above the selected confidence threshold.")
            else:
                data_list = []
                for box in boxes:
                    coords = [round(x, 2) for x in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]
                    
                    data_list.append({
                        "Object Class": cls_name,
                        "Confidence Score": f"{conf:.2%}",
                        "Bounding Box Coordinates": str(coords)
                    })
                
                st.table(data_list)

    # -------------------------------------------------------------------------
    # Tab 3: BLIP Transformer Captioning
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("BLIP Natural Image Captioning")
        
        with st.spinner("Generating caption..."):
            inputs = blip_processor(images=raw_image, return_tensors="pt").to(device)
            output = blip_model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=3,
                do_sample=False,
                repetition_penalty=1.4,
                length_penalty=1.2,
                early_stopping=True
            )
            caption = blip_processor.decode(output[0], skip_special_tokens=True)
            
        st.image(raw_image, caption="Target Image", width=500)
        st.success(f"**Generated Caption:** *{caption.capitalize()}*")

else:
    st.info("💡 Please upload an image from the sidebar to begin processing!")
