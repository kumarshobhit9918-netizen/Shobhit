# Breast Cancer Classification

## Machine Learning Model Evaluation Dashboard

This project is a Machine Learning based Breast Cancer Classification application developed using Python and Streamlit.

The application allows users to upload a test dataset, select a trained Machine Learning model, generate predictions, and evaluate the model using different performance metrics.

## Project Objective

The objective of this project is to classify breast cancer cases using multiple Machine Learning classification algorithms and compare their performance using standard evaluation metrics.

## Machine Learning Models

The following trained models are included in this project:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest

The trained models are stored in the `model` folder as `.pkl` files.

## Features

The Streamlit dashboard provides the following features:

1. Upload a CSV test dataset
2. Preview the uploaded dataset
3. Select a Machine Learning model
4. Generate predictions
5. Calculate evaluation metrics
6. Display the Confusion Matrix
7. Display the Classification Report

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC Score
- Confusion Matrix
- Classification Report

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── ML_Assignment_2.ipynb
├── test_data.csv
├── results_summary.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl