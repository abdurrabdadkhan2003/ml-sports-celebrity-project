# Sports Celebrity Image Classification

**End-to-end ML image classification app for recognizing sports celebrities using Python, OpenCV, and a Flask web interface.**

---

## Overview

This project is an end-to-end machine learning application that classifies images of sports celebrities into predefined categories through a web interface. It showcases the full lifecycle from raw image data, feature engineering, and model training to deployment as a simple browser-based app.

The system allows users to upload an image of a supported sports celebrity and returns the predicted class along with confidence scores for each class.

---

## Key Features

- Image classification of multiple sports celebrities (Messi, Federer, Kohli, Sharapova, Serena Williams)
- Complete ML pipeline: data cleaning, feature extraction, model training, evaluation, and inference
- Web-based UI built with HTML/CSS/JavaScript for uploading and classifying images
- Flask backend API for serving the trained model and handling image predictions
- Support for probability scores across all classes to interpret model confidence
- Modular project structure separating data, models, and web application components
- Easily extensible to new classes or updated models with minimal changes

---

## Project Artifacts / Deliverables

This repository includes:

- Jupyter notebooks for data exploration, feature extraction, and model training
- Trained model files and supporting assets (class mapping, feature extraction utilities)
- Flask application code (`app.py`) exposing a prediction endpoint
- Frontend files (HTML/CSS/JavaScript) for the image upload and results page
- Configuration files and utilities for preprocessing and inference
- Python-specific `.gitignore` to keep the repository clean

---

## Getting Started

### Prerequisites

- Python 3.8+ installed on your system
- Git for cloning the repository
- Recommended: virtual environment tool (venv or virtualenv)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdurrabdadkhan2003/ml-sports-celebrity-project.git
   cd ml-sports-celebrity-project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. **Open the web UI**
   - Navigate to `http://127.0.0.1:5000/` in your browser
   - Upload an image of a supported sports celebrity and view the prediction

---

## Technologies & Skills Demonstrated

**Technologies & Libraries**

- Python, NumPy, Pandas for data handling and preprocessing
- OpenCV for image processing and feature extraction
- Scikit-learn for model training, evaluation, and hyperparameter tuning
- Matplotlib / Seaborn for exploratory data analysis and visualization
- Flask for serving the ML model as a web API
- HTML, CSS, JavaScript for building the user-facing web interface
- Git & GitHub for version control and portfolio presentation

**Skills Demonstrated**

- End-to-end ML project structuring and reproducible workflows
- Image data preprocessing, feature engineering, and classical ML modeling
- Deployment of ML models via REST-style HTTP endpoints using Flask
- Basic frontend integration with a backend ML service
- Clean code organization and Python project best practices

---

## Results & Impact

- Achieved robust classification performance on a curated dataset of sports celebrity images
- Delivered an interactive web app that demonstrates ML capabilities to non-technical users
- Built an extensible codebase that can be adapted for other image classification use cases

*See PROJECT.md for detailed metrics, methodology, and technical implementation.*

---

## Future Enhancements

- Add support for more sports celebrities and additional training data
- Experiment with deep learning models (transfer learning with CNNs) for improved performance
- Deploy the app on a cloud platform (Render, Railway, Heroku) for public access
- Add user feedback logging to analyze common failure cases
- Integrate model monitoring and automated retraining with updated datasets
- Add batch processing for multiple image predictions

---

## About My Role

- Designed and implemented the complete ML pipeline from data preprocessing to model deployment
- Built and integrated the Flask backend with the web UI for a seamless user experience
- Managed version control, documentation, and project structure to align with professional portfolio standards

---

## Documentation

- **[PROJECT.md](./PROJECT.md)** - Detailed technical implementation, methodology, and results
- **[USAGE.md](./USAGE.md)** - Code examples and API usage (coming soon)
- **[SETUP.md](./SETUP.md)** - Detailed environment setup and troubleshooting (coming soon)

---

## License

MIT License - See LICENSE file for details

---

**Built with passion by [Abdurrab Dadkhan](https://github.com/abdurrabdadkhan2003)**
