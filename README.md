# 🚀 Satellite Change Detection System

## 📌 Overview

This project is an AI-powered web application that detects changes between two satellite images. It analyzes urban development by identifying variations in **buildings**, **vehicles**, and overall pixel differences, and then estimates a **threat level** based on detected changes.

The application is built using **YOLOv8**, **OpenCV**, and deployed with **Streamlit** for an interactive user experience.

---

## ✨ Features

* 🏢 Building detection using image processing techniques
* 🚗 Vehicle detection using YOLOv8 object detection
* 🔥 Pixel-level change detection (heatmap visualization)
* 📊 Automated comparison between two timeframes (T1 vs T2)
* ⚠️ Threat level classification (LOW / MEDIUM / HIGH)
* 📥 Downloadable CSV results
* 🌐 Simple web interface for easy usage

---

## 🧠 How It Works

1. User uploads two satellite images (T1 and T2)
2. Images are preprocessed using CLAHE for enhancement
3. YOLOv8 detects vehicles (cars, buses, trucks)
4. OpenCV detects buildings using contour analysis
5. Pixel-wise difference generates a change heatmap
6. System compares results and assigns a threat level

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Ultralytics YOLOv8
* OpenCV
* NumPy & Pandas
* Matplotlib

---

## 📂 Project Structure

```
project/
│── app.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the app

```
streamlit run app.py
```

---

## 🚀 Deployment

This app can be deployed easily using **Streamlit Cloud**:

1. Push your project to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. Select `app.py` and deploy

---

## 📊 Output

* Annotated images (T1 and T2)
* Change heatmap
* Metrics:

  * Building count difference
  * Vehicle count difference
  * Threat level
* Downloadable CSV report

---

## ⚠️ Limitations

* Building detection uses basic contour methods (not deep learning)
* Accuracy depends on image quality and resolution
* Runs on CPU in deployment (slower than GPU)

---

## 🔮 Future Improvements

* Use deep learning for building segmentation
* Integrate real satellite datasets
* Add geospatial map visualization
* Improve detection accuracy with custom-trained models

---

## 👩‍💻 Author

Your Name

---

## 📜 License

This project is open-source and available under the MIT License.
