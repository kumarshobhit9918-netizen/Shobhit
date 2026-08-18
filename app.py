import streamlit as st 
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification")
st.write("Machine Learning Model Evaluation Dashboard")
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
# Load trained models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl")
}

# Load the scaler
scaler = joblib.load("model/scaler.pkl")

st.success("All trained models loaded successfully!")
st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing test data",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.success("CSV uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(test_df.head())
    st.header("2. Select Machine Learning Model")

    selected_model_name = st.selectbox(
        "Choose a model:",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    st.write("Selected Model:", selected_model_name)
    st.header("3. Model Prediction")

    # Separate features and target
    X_new = test_df.drop("target", axis=1)
    y_new = test_df["target"]

    # Scale the input data
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    y_pred = selected_model.predict(X_new_scaled)

    st.success("Prediction completed successfully!")

    # Evaluation metrics
    accuracy = accuracy_score(y_new, y_pred)
    precision = precision_score(y_new, y_pred)
    recall = recall_score(y_new, y_pred)
    f1 = f1_score(y_new, y_pred)
    mcc = matthews_corrcoef(y_new, y_pred)

    try:
        y_prob = selected_model.predict_proba(X_new_scaled)[:, 1]
        auc = roc_auc_score(y_new, y_prob)
    except:
        auc = None

    st.header("4. Evaluation Results")

    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")
    st.metric("MCC", f"{mcc:.4f}")

    if auc is not None:
        st.metric("AUC", f"{auc:.4f}")
        st.header("5. Confusion Matrix")

cm = confusion_matrix(y_new, y_pred)

st.write("Confusion Matrix:")
st.write(cm)

st.header("6. Classification Report")

report = classification_report(y_new, y_pred)
st.text(report)
