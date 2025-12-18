# Sports Celebrity Image Classification – Project Details

## Problem Statement & Objectives

The goal of this project is to classify images of sports celebrities into predefined categories using machine learning techniques. The objective is to build an end-to-end system that can take an input image from a user and return the most likely celebrity label along with confidence scores.

Key objectives:

- Prepare and clean a labeled dataset of sports celebrity images.
- Engineer meaningful features from images and train a robust classification model.
- Expose the model through a user-friendly web application for real-time inference.

---

## Technical Approach / Methodology

- **Data Preparation:** Collected and organized images into class-wise folders and applied preprocessing such as resizing, color conversion, and noise reduction using OpenCV.
- **Feature Engineering:** Extracted features from images (e.g., raw pixel values, transformations, or handcrafted features as applicable to your implementation).
- **Modeling:** Trained and compared multiple classical ML models (e.g., logistic regression, SVM, or others) using scikit-learn to select the best-performing classifier.
- **Evaluation:** Evaluated models using train-test splits, confusion matrices, and standard metrics such as accuracy, precision, recall, and F1-score.
- **Deployment:** Wrapped the trained model in a Flask application with a simple HTML-based frontend for image upload and result visualization.

---

## Implementation Details

- Organized the codebase into logical modules for data preprocessing, feature extraction, model training, and prediction utilities.
- Stored class mappings and any preprocessing parameters so that training and inference pipelines remain consistent.
- Implemented a prediction endpoint in Flask that:
  - Accepts an uploaded image from the client.
  - Applies the same preprocessing and feature extraction pipeline.
  - Loads the trained model and returns predictions and class probabilities.
- The frontend (HTML/CSS/JS) sends the image to the backend and displays both the predicted class and probability distribution across all classes.

---

## Key Learnings & Challenges Solved

- Gained practical experience in handling image data, including challenges like varying resolutions, lighting conditions, and backgrounds.
- Learned how to design reproducible ML pipelines that keep preprocessing consistent between training and inference.
- Overcame integration challenges between the ML model and the Flask backend, particularly around file handling and performance.
- Improved understanding of serving ML models through web APIs and providing a smooth user experience via a simple UI.

---

## Tools & Libraries Used

- **Programming Language:** Python
- **Data & ML:** NumPy, Pandas, Scikit-learn
- **Image Processing:** OpenCV
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask for backend API
- **Frontend:** HTML, CSS, JavaScript for the browser UI
- **Version Control:** Git & GitHub for source code management and collaboration

---

## Results & Deliverables

- A trained sports celebrity image classification model with documented performance metrics (insert your specific scores).
- A working Flask-based web application that demonstrates real-time predictions from the model.
- Reproducible notebooks and scripts that cover data preparation, training, evaluation, and deployment steps.
- A portfolio-ready GitHub repository with structured documentation (README, PROJECT notes, and supporting files).

---

## Metrics & Performance

*To be updated with actual results:*

- **Overall Accuracy:** [Insert %]
- **Precision:** [Insert scores per class]
- **Recall:** [Insert scores per class]
- **F1-Score:** [Insert scores per class]
- **Confusion Matrix:** [Describe key insights]

---

## File Structure

```
ml-sports-celebrity-project/
├── README.md                 # Main project overview
├── PROJECT.md               # This file - detailed technical details
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
│
├── notebooks/              # Jupyter notebooks for analysis and training
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/                    # Source code
│   ├── preprocessing.py    # Image preprocessing utilities
│   ├── features.py         # Feature extraction functions
│   └── model.py            # Model training and evaluation
│
├── server/                 # Flask application
│   ├── app.py              # Flask app entry point
│   └── routes.py           # API routes
│
├── UI/                     # Frontend files
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── model/                  # Trained models and artifacts
│   ├── classifier.pkl      # Trained model file
│   └── class_mapping.json  # Class label mapping
│
└── images_dataset/         # Dataset (usually .gitignored)
    ├── messi/
    ├── federer/
    ├── kohli/
    ├── sharapova/
    └── serena/
```

---

## Future Improvements

- Migrate to deep learning models (CNN with transfer learning) for improved accuracy.
- Add real-time model performance monitoring and automated retraining.
- Support for more celebrity classes and continuous dataset expansion.
- Deploy on cloud platforms for global accessibility.
- Implement batch processing for handling multiple predictions.

---

**Last Updated:** December 2025
