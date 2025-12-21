# Sports Celebrity Image Classification

**An end-to-end machine learning project for classifying images of sports celebrities using classical ML, computer vision, and a Flask web interface.**

---

## 📋 Project Summary & Purpose

This project demonstrates a **complete ML pipeline** — from raw image data to a fully functional web application. It classifies images of famous sports personalities (Messi, Federer, Kohli, Sharapova, Serena Williams) using image processing techniques and machine learning models.

The final deliverable is an interactive web application where users can upload an image and instantly see which celebrity the model predicts, along with confidence scores.

---

## 🎯 Problem Statement

**Challenge:** Build a system that can automatically identify sports celebrities from images in real-time, with high accuracy and confidence metrics.

**Approach:** 
- Collect and preprocess a labeled dataset of sports celebrity images
- Extract meaningful features using Haar Cascade face detection and image processing
- Train and evaluate multiple classical ML models
- Deploy the best model through a user-friendly Flask web application

---

## 🤖 Machine Learning Approach

### Pipeline Overview

```
Raw Images → Preprocessing → Feature Extraction → Model Training → Evaluation → Deployment
```

### Key Steps

1. **Data Preparation**
   - Organized labeled dataset into class-wise folders
   - Applied preprocessing: resizing, color conversion, noise reduction using OpenCV
   - Handled class imbalance with data augmentation

2. **Feature Engineering**
   - Used Haar Cascade classifiers to detect faces and eyes
   - Extracted region of interest (ROI) from detected faces
   - Flattened and normalized pixel values for model input
   - Created feature vectors: raw pixels + optional handcrafted features

3. **Model Selection & Training**
   - Trained multiple classifiers:
     - **Logistic Regression**: ~78.7% accuracy ⭐ (Best performer)
     - **SVM (Support Vector Machine)**: ~74.5% accuracy
     - **Random Forest**: ~63.8% accuracy
   - Used GridSearchCV for hyperparameter tuning
   - Performed train-test split (80-20) for validation

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix analysis
   - Cross-validation for robustness

---

## 📊 Dataset Used & Features

### Dataset Composition

| Celebrity | Class ID | Samples |
|-----------|----------|----------|
| Lionel Messi | 0 | ~40 images |
| Roger Federer | 1 | ~40 images |
| Virat Kohli | 2 | ~40 images |
| Maria Sharapova | 3 | ~40 images |
| Serena Williams | 4 | ~40 images |

**Total:** ~200 labeled images

### Data Preprocessing

- **Image Resizing:** Standardized to 32×32 pixels per analysis stage
- **Color Space Conversion:** RGB to Grayscale for feature extraction
- **Face Detection:** Haar Cascade to isolate facial regions
- **Noise Handling:** Removed images without detected faces

### Feature Extraction

```python
# Feature vector per image:
- Raw pixel values from detected face ROI (32×32 = 1024 features)
- Normalized to [0, 1] range
- Optional: Wavelet transforms for additional texture features
```

---

## 📈 Training Details & Results

### Model Performance Summary

```
┌─────────────────────┬──────────┬───────────┬─────────┐
│ Model               │ Accuracy │ Precision │ Recall  │
├─────────────────────┼──────────┼───────────┼─────────┤
│ Logistic Regression │ 78.7%    │ 0.79      │ 0.79    │
│ SVM                 │ 74.5%    │ 0.74      │ 0.74    │
│ Random Forest       │ 63.8%    │ 0.64      │ 0.64    │
└─────────────────────┴──────────┴───────────┴─────────┘
```

### Confusion Matrix (Logistic Regression - Best Model)

```
                Predicted
                Messi  Federer  Kohli  Sharapova  Serena
Actual Messi      8      0        0       0          2
       Federer    1      5        0       1          0
       Kohli      1      1        3       1          0
       Sharapova  0      0        0       7          0
       Serena     0      0        0       0          8
```

### Key Insights

- **Logistic Regression performed best** due to the linear separability of feature space after processing
- **Serena & Sharapova**: Near-perfect classification (100%)
- **Messi**: Slight confusion with other classes (scoring 80%)
- **Overall weighted F1-Score: 0.787**

---

## 🚀 Instructions to Run & Evaluate

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdurrabdadkhan2003/ml-sports-celebrity-project.git
   cd ml-sports-celebrity-project
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python server/server.py
   ```
   The server will start on `http://127.0.0.1:5000`

2. **Access the web interface**
   - Open your browser and navigate to `http://127.0.0.1:5000`
   - Click "Choose File" and select an image
   - Click "Upload" to get predictions
   - View the predicted celebrity and confidence scores

### Running Training & Evaluation

1. **Execute the Jupyter notebook**
   ```bash
   jupyter notebook model/sports_celebrity_classification.ipynb
   ```
   This notebook contains:
   - Data exploration and visualization
   - Feature extraction pipeline
   - Model training and comparison
   - Evaluation metrics and confusion matrix

2. **Train a custom model (optional)**
   - Modify the `model/sports_celebrity_classification.ipynb`
   - Update hyperparameters as needed
   - Export the trained model to `server/artifacts/saved_model.pkl`

### Project Structure

```
ml-sports-celebrity-project/
├── README.md                              # Project overview (this file)
├── PROJECT.md                             # Detailed technical documentation
├── requirements.txt                       # Python dependencies
│
├── model/
│   ├── sports_celebrity_classification.ipynb    # Main training notebook
│   ├── data_cleaning.ipynb                      # Data preprocessing
│   ├── dataset/                                 # Training dataset (class folders)
│   │   ├── lionel_messi/
│   │   ├── roger_federer/
│   │   ├── virat_kohli/
│   │   ├── maria_sharapova/
│   │   └── serena_williams/
│   ├── test_images/                            # Test images for evaluation
│   ├── class_dictionary.json                   # Class name mappings
│   ├── saved_model.pkl                         # Trained model (exported)
│   └── opencv/haarcascades/                    # Haar Cascade XML files
│
├── server/
│   ├── server.py                          # Flask application entry point
│   ├── util.py                            # Prediction utilities & preprocessing
│   ├── wavelet.py                         # Wavelet feature extraction
│   ├── artifacts/
│   │   ├── saved_model.pkl                # Model artifact for inference
│   │   └── class_dictionary.json          # Class mappings
│   ├── test_images/                       # Test images for API
│   └── haarcascades/                      # OpenCV Haar Cascades
│
├── UI/
│   ├── app.html                           # Web interface (HTML)
│   ├── app.css                            # Styling
│   ├── app.js                             # Frontend logic
│   ├── images/                            # UI assets & sample images
│   └── test_images/                       # Test images for upload
│
└── docs/
    └── README-*.md                        # Additional documentation
```

### Testing & Evaluation

**Option 1: Web Interface (Recommended for quick testing)**
1. Start the server (see above)
2. Use the web interface to test with images
3. Observe predictions and confidence scores in real-time

**Option 2: Programmatic Testing**
```python
from server.util import classify_image

# Test prediction
image_path = "path/to/test/image.jpg"
result = classify_image(image_path)
print(f"Predicted: {result['class']}")
print(f"Confidence: {result['class_probability']}")
```

**Option 3: Jupyter Notebook**
- Run cells in `model/sports_celebrity_classification.ipynb`
- Review training metrics and visualizations
- Generate confusion matrix and classification reports

---

## 🛠️ Technologies & Tools Used

### Machine Learning & Data Processing
- **Python 3.8+** — Core programming language
- **NumPy & Pandas** — Numerical computing and data manipulation
- **Scikit-learn** — Model training, evaluation, and hyperparameter tuning
- **OpenCV** — Image processing and Haar Cascade face detection
- **Jupyter Notebook** — Interactive data exploration and model development

### Web Framework & Deployment
- **Flask** — Lightweight web framework for API and interface serving
- **HTML/CSS/JavaScript** — Frontend user interface

### Visualization & Analysis
- **Matplotlib & Seaborn** — Charts, plots, and confusion matrices

### Version Control
- **Git & GitHub** — Repository management and collaboration

---

## 💡 Key Skills Demonstrated

✅ **End-to-End ML Pipeline:**  Data → Preprocessing → Training → Evaluation → Deployment

✅ **Computer Vision:** Face detection using Haar Cascades, ROI extraction, image preprocessing

✅ **Feature Engineering:** Handcrafted features, normalization, dimensionality handling

✅ **Model Evaluation:** Confusion matrix, precision-recall-F1, cross-validation, GridSearch

✅ **Web Deployment:** Flask API, form handling, file uploads, JSON responses

✅ **Code Quality:** Modular structure, clear documentation, reproducible experiments

---

## 🚧 Future Enhancements

1. **Deep Learning Migration**
   - Implement CNN using PyTorch/TensorFlow
   - Use transfer learning (ResNet, VGG, EfficientNet)
   - Expected accuracy improvement: 85%+ with larger datasets

2. **Scalability**
   - Add more celebrity classes (50+ personalities)
   - Implement batch prediction API
   - Containerize with Docker for easy deployment

3. **Robustness**
   - Data augmentation (rotation, flipping, brightness adjustment)
   - Model ensemble for improved confidence
   - Confidence threshold filtering

4. **Production Readiness**
   - Deploy on cloud (AWS, GCP, Azure)
   - Add logging and monitoring
   - Implement A/B testing for model updates
   - Create automated retraining pipeline

5. **Advanced Features**
   - Real-time video stream classification
   - Face similarity matching
   - Integration with face recognition APIs

---

## 📚 Documentation

- **[PROJECT.md](PROJECT.md)** — Technical deep-dive: methodology, implementation details, and challenges
- **[model/sports_celebrity_classification.ipynb](model/sports_celebrity_classification.ipynb)** — Training notebook with step-by-step explanations
- **Code Comments** — Inline documentation in `server/util.py` and `server/server.py`

---

## 👨‍💼 Author

**Abdurrab Dadkhan** — ML Engineer & Data Science Enthusiast

- 🔗 [GitHub](https://github.com/abdurrabdadkhan2003)
- 📧 Open to collaborations and feedback!

---

## 📄 License

MIT License — See LICENSE file for details

---

## ⭐ If You Found This Helpful

Consider giving this repository a **star** ⭐ if you found it useful for learning ML, web deployment, or building your portfolio!

Happy coding! 🎉
