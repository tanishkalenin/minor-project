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
def load_image(file):
    bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(bytes_data, 1)

# ==========================================
# PREPROCESS
# ==========================================
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

# ==========================================
# CHANGE MASK
# ==========================================
def get_change_mask(img1, img2):
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # clean noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

# ==========================================
# VEHICLE CHANGE (ROBUST)
# ==========================================
def detect_vehicle_changes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_v = []
    removed_v = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h

        # vehicle-like size (tuneable)
        if 50 < area < 1500:
            new_v.append((x,y,w,h))

        elif area < 50:
            removed_v.append((x,y,w,h))

    return new_v, removed_v

# ==========================================
# BUILDING CHANGE
# ==========================================
def detect_buildings(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,2
    )

    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    changed = []
    all_b = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h

        if 500 < area < 8000:
            all_b.append((x,y,w,h))

            region = mask[y:y+h, x:x+w]
            if np.sum(region) > 800:
                changed.append((x,y,w,h))

    return all_b, changed

# ==========================================
# INTELLIGENCE
# ==========================================
def intelligence(b, nv, rv):
    if b > 5 or nv > 5:
        return "HIGH"
    elif nv > 0 or rv > 0:
        return "MEDIUM"
    else:
        return "LOW"

# ==========================================
# DRAW
# ==========================================
def draw(img, buildings, changed_b, new_v, removed_v):
    out = img.copy()

    # buildings
    for x,y,w,h in buildings:
        cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,0),1)

    # changed buildings
    for x,y,w,h in changed_b:
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)

    # new vehicles
    for x,y,w,h in new_v:
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,255),2)

    # removed vehicles
    for x,y,w,h in removed_v:
        cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,255),2)

    return out

# ==========================================
# RUN
# ==========================================
if uploaded_t1 and uploaded_t2:

    img1 = preprocess(load_image(uploaded_t1))
    img2 = preprocess(load_image(uploaded_t2))

    if st.button("Run Analysis"):

        mask = get_change_mask(img1, img2)

        # buildings
        all_b, changed_b = detect_buildings(img2, mask)

        # vehicles (robust)
        new_v, removed_v = detect_vehicle_changes(mask)

        # intelligence
        threat = intelligence(len(changed_b), len(new_v), len(removed_v))

        # draw
        out1 = draw(img1, all_b, changed_b, new_v, removed_v)
        out2 = draw(img2, all_b, changed_b, new_v, removed_v)

        # DISPLAY
        st.subheader("Results")

        c1, c2 = st.columns(2)
        c1.image(out1, caption="T1")
        c2.image(out2, caption="T2")

        st.image(mask, caption="Change Map")

        st.metric("Changed Buildings", len(changed_b))
        st.metric("New Vehicles", len(new_v))
        st.metric("Removed Vehicles", len(removed_v))
        st.metric("Threat Level", threat)

        # CSV
        df = pd.DataFrame([
            {"metric":"changed_buildings","value":len(changed_b)},
            {"metric":"new_vehicles","value":len(new_v)},
            {"metric":"removed_vehicles","value":len(removed_v)},
            {"metric":"threat","value":threat}
        ])

        st.download_button("Download CSV", df.to_csv(index=False), "results.csv")
