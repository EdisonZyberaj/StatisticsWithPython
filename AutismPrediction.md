#  Autism Prediction using Machine Learning & Neural Networks

This project focuses on predicting Autism Spectrum Disorder (ASD) using both traditional machine learning models and a neural network built with PyTorch.

---

##  Project Overview

Autism Spectrum Disorder (ASD) is a developmental condition that affects communication and behavior. Early detection is crucial, and this project builds predictive models using screening and demographic data.

The project implements a full data science and AI pipeline:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Neural network implementation

---

##  Technologies Used

- Python 🐍
- Jupyter Notebook
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- PyTorch 🔥

---

## Dataset

The dataset includes:
- Behavioral screening scores (A1–A10)
- Age
- Gender
- Ethnicity
- Family history of autism
- Medical indicators (e.g., jaundice)
- Screening results

---

##  Workflow

### 1. Data Preprocessing
- Handling categorical variables using Label Encoding
- Removing unnecessary columns
- Handling outliers using IQR method
- Feature scaling (for Neural Network)

### 2. Handling Imbalanced Data
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**

### 3. Model Training
- Train-test split (80/20)
- Training multiple models

### 4. Model Evaluation
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- Cohen Kappa Score

---

## Algorithms Used

### Machine Learning Models
- Decision Tree Classifier
- Random Forest Classifier

### Deep Learning Model
- Neural Network (PyTorch)
  - Architecture: 19 → 64 → 32 → 1
  - Activation: ReLU
  - Dropout: 0.3
  - Loss Function: Binary Cross Entropy (BCEWithLogitsLoss)
  - Optimizer: Adam

---


 **Best Model:** Random Forest  
Neural Network performed competitively with strong generalization.


1. Clone the repository:
```bash
git clone https://github.com/EdisonZyberaj/StatisticsWithPython.git