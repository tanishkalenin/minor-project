import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==========================================
# UI
# ==========================================
st.title("🚀 Satellite Change Detection System")

uploaded_t1 = st.file_uploader("Upload T1 Image", type=["jpg", "png"])
uploaded_t2 = st.file_uploader("Upload T2 Image", type=["jpg", "png"])

# ==========================================
# IMAGE LOADER
# ==========================================
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

# ==========================================
# PREPROCESS
# ==========================================
def preprocess(img):
    img = cv2.resize(img, (640, 640))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)

    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img

# ==========================================
# VEHICLE DETECTION
# ==========================================
def detect_vehicles(img):
    results = model(img, conf=0.4, imgsz=320)
    count = 0
    boxes = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            label = r.names[cls]

            if label in ["car", "bus", "truck"]:
                count += 1
                boxes.append(b.xyxy[0].tolist())

    return count, boxes

# ==========================================
# BUILDING DETECTION
# ==========================================
def detect_buildings(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    buildings = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 100 < w * h < 5000:
            buildings.append((x, y, w, h))

    return len(buildings), buildings

# ==========================================
# CHANGE DETECTION
# ==========================================
def pixel_change(img1, img2):
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return thresh

# ==========================================
# INTELLIGENCE
# ==========================================
def intelligence(b1, b2, v1, v2):
    if b2 > b1:
        return "HIGH"
    elif v2 > v1:
        return "MEDIUM"
    else:
        return "LOW"

# ==========================================
# DRAW RESULTS
# ==========================================
def draw(img, vehicles, buildings):
    out = img.copy()

    for box in vehicles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for (x, y, w, h) in buildings:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return out

# ==========================================
# RUN
# ==========================================
if uploaded_t1 and uploaded_t2:

    img1 = load_image(uploaded_t1)
    img2 = load_image(uploaded_t2)

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    if st.button("Run Analysis"):

        v1, vb1 = detect_vehicles(img1)
        v2, vb2 = detect_vehicles(img2)

        b1, bb1 = detect_buildings(img1)
        b2, bb2 = detect_buildings(img2)

        heatmap = pixel_change(img1, img2)
        threat = intelligence(b1, b2, v1, v2)

        out1 = draw(img1, vb1, bb1)
        out2 = draw(img2, vb2, bb2)

        # ==========================================
        # DISPLAY
        # ==========================================
        st.subheader("Results")

        col1, col2 = st.columns(2)
        col1.image(out1, caption="T1")
        col2.image(out2, caption="T2")

        st.image(heatmap, caption="Change Map")

        st.metric("Buildings Change", b2 - b1)
        st.metric("Vehicle Change", v2 - v1)
        st.metric("Threat Level", threat)

        # ==========================================
        # CSV DOWNLOAD
        # ==========================================
        df = pd.DataFrame([
            {"type": "building", "t1": b1, "t2": b2},
            {"type": "vehicle", "t1": v1, "t2": v2}
        ])

        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            "results.csv",
            "text/csv"
        )
