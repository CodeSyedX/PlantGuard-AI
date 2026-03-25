# 🌿 PlantGuard AI: Multi-Class Plant Disease Diagnostic System

## 📌 Project Overview
PlantGuard AI is an end-to-end Deep Learning application designed to help farmers and agricultural experts identify plant diseases early using leaf images.

The system uses a Convolutional Neural Network (CNN) to classify plant leaf images into 38 different categories across multiple plant species such as Apple, Corn, Grape, Potato, and Tomato.

The application provides instant disease diagnosis and treatment recommendations through a simple web interface.

---

## 🎯 Objective
The goal of this project is to bridge the gap between advanced computer vision models and real-world agricultural applications by creating a user-friendly platform for fast and accurate plant disease detection.

---

## 🚀 Features

⚡ Instant Diagnosis  
Upload a leaf image and receive predictions in under 2 seconds.

🌾 High Coverage  
Supports 38 plant-disease classes including healthy leaves.

🖥️ Interactive Web Interface  
Built using Streamlit with a clean and responsive layout.

📊 Confidence Visualization  
Displays prediction confidence using progress bars.

💡 Treatment Recommendations  
Provides brief suggestions to help manage detected diseases.

🚀 Optimized Performance  
Uses `@st.cache_resource` for efficient model loading.

---

## 🏗️ Technical Architecture

The project follows a typical Deep Learning workflow:

### Data Acquisition
Dataset used: PlantVillage Dataset  
Total images: 54,000+  
Source: Kaggle

### Data Preprocessing
Image resizing: 224 × 224  
Normalization: 1/255  
Data augmentation used to improve model generalization.

### Model Architecture

Sequential CNN consisting of:

Conv Layer 1 – 32 filters  
Conv Layer 2 – 64 filters  
Conv Layer 3 – 128 filters  
MaxPooling Layers  
Dropout (0.5) to prevent overfitting  
Dense Layer (256 neurons)  
Softmax Output Layer (38 classes)

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### 2️⃣ Install Dependencies

Make sure Python 3.9 or later is installed.

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

After running the command, open the local Streamlit URL displayed in your terminal.

---

## 📊 Dataset Information

The model was trained using the PlantVillage Dataset, which contains images of healthy and diseased crop leaves.

### Plants Covered

Apple  
Apple Scab  
Black Rot  
Cedar Apple Rust  
Healthy

Tomato  
Bacterial Spot  
Early Blight  
Late Blight  
Leaf Mold  
Mosaic Virus  
Healthy

Potato  
Early Blight  
Late Blight  
Healthy

Grape  
Black Rot  
Esca (Black Measles)  
Leaf Blight  
Healthy

and many more plant-disease classes.

---

## 📷 Example Workflow

1️⃣ Upload a leaf image  
2️⃣ The model processes the image  
3️⃣ CNN predicts the disease class  
4️⃣ The system displays:

Predicted Disease  
Confidence Score  
Suggested Treatment

---

## 💻 Technologies Used

Python  
TensorFlow / Keras  
Streamlit  
NumPy  
OpenCV  
Matplotlib

---

## 👨‍💻 Developer

Syed Aafreen  
Computer Science & Engineering  
Vellore Institute of Technology (VIT)

---

## ⭐ Future Improvements

Mobile-friendly interface  
Real-time camera detection  
Integration with agricultural advisory systems  
Multi-language support for farmers

---

## 📜 License

This project is for educational and research purposes.