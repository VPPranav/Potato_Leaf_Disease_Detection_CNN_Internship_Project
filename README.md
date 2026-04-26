# 🥔 Potato Leaf Disease Detection System

> An AI-powered web application that detects potato leaf diseases from images using a Convolutional Neural Network (CNN), built with TensorFlow and deployed via Streamlit.

**Developed by Pranav V P · Internship Project 2026**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Performance Metrics](#-performance-metrics)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Disease Classes](#-disease-classes)
- [Key Fix — Preprocessing Bug](#-key-fix--preprocessing-bug)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 🌿 Overview

Potato is one of the world's most important staple crops, yet it is highly susceptible to fungal diseases that can devastate entire harvests if left undetected. This project builds an end-to-end deep learning solution that allows farmers, agronomists, and agricultural students to upload a photograph of a potato leaf and receive an instant AI-generated diagnosis, complete with confidence scores, disease descriptions, and recommended treatment actions.

The system classifies potato leaves into three categories:

| Class | Description |
|---|---|
| 🟠 Early Blight | Caused by *Alternaria solani* fungus |
| 🔴 Late Blight | Caused by *Phytophthora infestans* oomycete |
| 🟢 Healthy | No disease detected |

---

## 🧩 Problem Statement

Traditional disease detection in potato crops relies on manual inspection by trained agricultural experts — a process that is:

- ⏱️ **Time-consuming** on large farms
- 💰 **Expensive** due to expert consultation costs
- ❌ **Inaccessible** in remote farming communities
- 🔁 **Reactive** rather than preventive

Early and Late Blight together are responsible for significant annual crop losses worldwide. Automated image-based diagnosis using deep learning enables **early detection**, **fast response**, and **scalable deployment** across devices with a simple browser or smartphone.

---

## 🚀 Live Demo

To run locally:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## ✨ Features

- 📤 **Drag-and-drop image upload** — supports JPG, JPEG, PNG formats
- 🔬 **Instant AI diagnosis** — prediction in under a second
- 📊 **Full probability breakdown** — confidence scores for all three classes
- 💊 **Actionable recommendations** — specific treatment advice per diagnosis
- 📋 **Disease reference guide** — embedded educational content
- 🎨 **Professional UI** — custom CSS with Google Fonts, animated cards, gradient themes
- ⚡ **Model caching** — `@st.cache_resource` ensures the model loads only once
- 📱 **Responsive layout** — works on desktop and tablet browsers

---

## 📁 Project Structure

```
potato-disease-detection/
│
├── app.py                                      # Streamlit web application
├── potato_disease_model.keras                  # Trained CNN model (Keras format)
├── Potato_Leaf_Disease_Detection_CNN_Notebook.ipynb  # Training notebook
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| **Source** | [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/hafiznouman786/potato-plant-diseases-data) |
| **Downloaded via** | `kagglehub` |
| **Classes** | Early Blight, Late Blight, Healthy |
| **Image format** | RGB JPEG |
| **Input size** | Resized to 256 × 256 pixels |
| **Dataset split** | 80% Train · 10% Validation · 10% Test |
| **Loader** | `tf.keras.preprocessing.image_dataset_from_directory()` |

> **Note on class ordering:** `image_dataset_from_directory` sorts folder names alphabetically, producing the class index order: `Early Blight (0) → Late Blight (1) → Healthy (2)`. This exact order is preserved in `CLASS_NAMES` in `app.py`.

---

## 🧠 Model Architecture

The model is a custom Sequential CNN built with TensorFlow/Keras:

```
Input (256, 256, 3)
    │
    ├── Data Augmentation
    │       ├── RandomFlip("horizontal_and_vertical")
    │       └── RandomRotation(0.2)
    │
    ├── Rescaling(1./255)           ← Normalisation built into the model
    │
    ├── Conv2D(32, 3×3, ReLU)
    ├── MaxPooling2D(2×2)
    │
    ├── Conv2D(64, 3×3, ReLU)
    ├── MaxPooling2D(2×2)
    │
    ├── Conv2D(64, 3×3, ReLU)
    ├── MaxPooling2D(2×2)
    │
    ├── Flatten
    ├── Dense(64, ReLU)
    └── Dense(3, Softmax)           ← Output: probability for each class
```

**Compilation settings:**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Loss function | `SparseCategoricalCrossentropy(from_logits=False)` |
| Metric | Accuracy |
| Epochs | 15 |
| Batch size | 32 |

---

## 🔄 Training Pipeline

1. **Dataset download** via KaggleHub
2. **Dataset loading** with `image_dataset_from_directory` (shuffle enabled)
3. **Train/Val/Test split** — 80% / 10% / 10% using `.take()` and `.skip()`
4. **Performance optimisation** — `.cache().shuffle(1000).prefetch(AUTOTUNE)` on all splits
5. **Data augmentation** — random flip and rotation baked into the model graph
6. **Model training** — 15 epochs with validation monitoring
7. **Evaluation** — confusion matrix, classification report (precision/recall/F1), ROC-AUC curves
8. **Model saving** — exported as `potato_disease_model.keras`

---

## 📈 Performance Metrics

The model was evaluated on the held-out test set:

| Metric | Value |
|---|---|
| **Test Accuracy** | Reported in notebook |
| **Classification Report** | Precision · Recall · F1 per class |
| **Confusion Matrix** | Normalised heatmap (Seaborn) |
| **ROC-AUC** | Computed per class (multiclass OvR) |
| **Precision-Recall AUC** | Computed per class |

> Run all evaluation cells in `Potato_Leaf_Disease_Detection_CNN_Notebook.ipynb` to reproduce the full metrics report.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1 — Clone or download the project

```bash
git clone https://github.com/your-username/potato-disease-detection.git
cd potato-disease-detection
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
streamlit>=1.35.0
tensorflow>=2.15.0
numpy>=1.24.0
Pillow>=10.0.0
```

### Step 4 — Place the model file

Ensure `potato_disease_model.keras` is in the **same directory** as `app.py`:

```
potato-disease-detection/
├── app.py
├── potato_disease_model.keras   ← required here
└── requirements.txt
```

---

## 💻 Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

### Steps to use the app

1. Click **Browse files** or drag a potato leaf image (JPG/PNG) into the upload zone
2. The model runs inference automatically
3. View the **diagnosis result** — class name and confidence percentage
4. Review the **probability breakdown** for all three classes
5. Read the **recommended action** for treatment or prevention
6. Consult the **Disease Reference Guide** for background information

---

## ☁️ Deployment

### Streamlit Community Cloud (free)

1. Push your project to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select the repository
4. Set `app.py` as the main file
5. Add `potato_disease_model.keras` to the repo (if size permits) or use Git LFS

> **Model size note:** Keras `.keras` files can be large. If the file exceeds GitHub's 100 MB limit, use [Git LFS](https://git-lfs.com/) or host the model on Google Drive / HuggingFace Hub and download it at startup.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t potato-disease-app .
docker run -p 8501:8501 potato-disease-app
```

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow 2.x · Keras |
| **Web Framework** | Streamlit |
| **Image Processing** | Pillow (PIL) |
| **Numerical Computing** | NumPy |
| **Evaluation** | scikit-learn (confusion matrix, ROC, PR curves) |
| **Visualisation** | Matplotlib · Seaborn |
| **Dataset Access** | KaggleHub |
| **Model Format** | Native Keras (`.keras`) |
| **Styling** | Custom CSS · Google Fonts (Playfair Display, DM Sans) |

---

## 🌿 Disease Classes

### 🟠 Early Blight — *Alternaria solani*

- **Appearance:** Dark brown concentric rings forming a target-board pattern, surrounded by yellow halo; typically appears on older, lower leaves first
- **Severity:** Moderate
- **Spread:** Airborne fungal spores; favoured by warm, humid conditions
- **Treatment:** Apply copper-based or chlorothalonil fungicide; remove and destroy infected leaves; practice crop rotation; avoid wetting foliage during irrigation

### 🔴 Late Blight — *Phytophthora infestans*

- **Appearance:** Dark, water-soaked lesions that rapidly turn brown/black; white fuzzy sporulation on leaf undersides under humid conditions
- **Severity:** Severe — can destroy an entire crop in days
- **Historical impact:** Caused the Great Irish Potato Famine (1845–1852)
- **Treatment:** Apply systemic fungicides (e.g., mefenoxam-based) immediately; remove and bag infected plant material; avoid overhead irrigation; plant resistant varieties

### 🟢 Healthy

- **Appearance:** Uniform bright green colouration, no lesions, spots, or wilting
- **Action:** Continue regular monitoring; maintain balanced fertilisation (NPK); ensure adequate drainage

---

## 🔧 Key Fix — Preprocessing Bug

The original `app.py` contained a **double-normalisation bug** that caused the model to always predict "Early Blight" regardless of the input image.

**Root cause:**

The CNN model contains a `Rescaling(1./255)` layer as part of its internal graph. The old preprocessing code also divided pixel values by 255 manually before feeding the image to the model:

```python
# ❌ WRONG — original code (double normalisation)
image = np.array(image) / 255.0   # pixel range becomes 0.0 – 1.0
# model then rescales again → pixel range becomes 0.0 – 0.004
```

This collapsed all pixel values to near-zero, destroying the input signal and biasing every prediction toward the first class.

**Fix applied in this version:**

```python
# ✅ CORRECT — fixed code
def preprocess_image(pil_image):
    img = pil_image.resize((256, 256))
    img_array = np.array(img)              # uint8, range 0–255
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    return img_array
    # The model's internal Rescaling layer handles normalisation
```

---

## 🔮 Future Improvements

- [ ] **More disease classes** — extend to other potato diseases (Blackleg, Common Scab, etc.)
- [ ] **Explainability** — add Grad-CAM heatmaps to highlight the regions driving predictions
- [ ] **Mobile app** — convert model to TensorFlow Lite for on-device Android/iOS inference
- [ ] **Multi-crop support** — generalise to tomato, maize, and other PlantVillage crops
- [ ] **Severity scoring** — estimate percentage leaf area affected
- [ ] **Batch processing** — allow upload and analysis of multiple images at once
- [ ] **REST API** — wrap the model in a FastAPI endpoint for third-party integrations
- [ ] **Offline mode** — Progressive Web App (PWA) with cached model for field use without internet

---

## 🙏 Acknowledgements

- **PlantVillage Dataset** — Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*
- **TensorFlow / Keras** — Google Brain Team
- **Streamlit** — Streamlit Inc.
- **Kaggle / KaggleHub** — for dataset hosting and easy programmatic access

---

## 👤 Author

**Pranav V P**
Internship Project · 2026

> *This project was developed as part of an AI/ML internship to demonstrate practical application of deep learning for real-world agricultural problems.*

---

<div align="center">

**🥔 Potato Leaf Disease Detection System**

Made with ❤️ using TensorFlow & Streamlit

</div>