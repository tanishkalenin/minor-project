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
# LOAD IMAGE
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
    boxes = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            label = r.names[cls]

            if label in ["car", "bus", "truck"]:
                boxes.append(b.xyxy[0].tolist())

    return boxes

# ==========================================
# VEHICLE CHANGE (FIXED LIKE BUILDINGS)
# ==========================================
def detect_vehicle_changes(img1, img2, boxes1, boxes2):
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, change_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    new_vehicles = []
    removed_vehicles = []

    # NEW vehicles (in T2)
    for box in boxes2:
        x1, y1, x2, y2 = map(int, box)
        region = change_mask[y1:y2, x1:x2]

        if region.size > 0 and np.sum(region) > 500:
            new_vehicles.append(box)

    # REMOVED vehicles (in T1)
    for box in boxes1:
        x1, y1, x2, y2 = map(int, box)
        region = change_mask[y1:y2, x1:x2]

        if region.size > 0 and np.sum(region) > 500:
            removed_vehicles.append(box)

    return new_vehicles, removed_vehicles

# ==========================================
# BUILDING CHANGE
# ==========================================
def detect_changed_buildings(img1, img2):
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, change_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    all_buildings = []
    changed_buildings = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if 100 < w * h < 5000:
            all_buildings.append((x, y, w, h))

            region = change_mask[y:y+h, x:x+w]

            if np.sum(region) > 500:
                changed_buildings.append((x, y, w, h))

    return len(all_buildings), len(changed_buildings), all_buildings, changed_buildings

# ==========================================
# CHANGE MAP
# ==========================================
def pixel_change(img1, img2):
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return thresh

# ==========================================
# INTELLIGENCE
# ==========================================
def intelligence(b_changed, new_v, removed_v):
    if b_changed > 5 or new_v > 5:
        return "HIGH"
    elif new_v > 0 or removed_v > 0:
        return "MEDIUM"
    else:
        return "LOW"

# ==========================================
# DRAW
# ==========================================
def draw(img, vehicles, new_v, removed_v, buildings, changed_buildings):
    out = img.copy()

    # vehicles normal
    for box in vehicles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # new vehicles
    for box in new_v:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # removed vehicles
    for box in removed_v:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # buildings
    for (x, y, w, h) in buildings:
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 1)

    # changed buildings
    for (x, y, w, h) in changed_buildings:
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return out

# ==========================================
# RUN
# ==========================================
if uploaded_t1 and uploaded_t2:

    img1 = preprocess(load_image(uploaded_t1))
    img2 = preprocess(load_image(uploaded_t2))

    if st.button("Run Analysis"):

        # vehicles
        boxes1 = detect_vehicles(img1)
        boxes2 = detect_vehicles(img2)

        new_v, removed_v = detect_vehicle_changes(img1, img2, boxes1, boxes2)

        # buildings
        b_total, b_changed, bb_all, bb_changed = detect_changed_buildings(img1, img2)

        # heatmap
        heatmap = pixel_change(img1, img2)

        # intelligence
        threat = intelligence(b_changed, len(new_v), len(removed_v))

        # draw
        out1 = draw(img1, boxes1, new_v, removed_v, bb_all, bb_changed)
        out2 = draw(img2, boxes2, new_v, removed_v, bb_all, bb_changed)

        # DISPLAY
        st.subheader("Results")

        col1, col2 = st.columns(2)
        col1.image(out1, caption="T1")
        col2.image(out2, caption="T2")

        st.image(heatmap, caption="Change Map")

        st.metric("Changed Buildings", b_changed)
        st.metric("New Vehicles", len(new_v))
        st.metric("Removed Vehicles", len(removed_v))
        st.metric("Threat Level", threat)

        # CSV
        df = pd.DataFrame([
            {"metric": "changed_buildings", "value": b_changed},
            {"metric": "new_vehicles", "value": len(new_v)},
            {"metric": "removed_vehicles", "value": len(removed_v)},
            {"metric": "threat_level", "value": threat}
        ])

        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            "results.csv",
            "text/csv"
        )
