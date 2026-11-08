
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

st.title("Dry Bean Classification")
st.write(
    "Evaluate five machine learning classifiers "
    "on the uploaded Dry Bean test dataset."
)

# Load saved models
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.pkl"
        ),
        "KNN": joblib.load(
            MODEL_DIR / "knn.pkl"
        ),
        "Naive Bayes": joblib.load(
            MODEL_DIR / "naive_bayes.pkl"
        ),
        "SVM": joblib.load(
            MODEL_DIR / "svm.pkl"
        )
    }

    features = joblib.load(
        MODEL_DIR / "feature_names.pkl"
    )

    return models, features


models, feature_names = load_models()

# Upload test dataset
st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 16 features and Class column",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    # Validate columns
    required_columns = feature_names + ["Class"]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")

    elif data.empty:
        st.error("The uploaded CSV is empty.")

    else:
        X = data[feature_names]
        y_true = data["Class"]

        if y_true.isna().any():
            st.error("The Class column contains missing values.")

        else:
            st.success(
                f"Dataset loaded successfully: {len(data)} rows"
            )

            # Evaluate all five models
            results = []
            predictions = {}

            for model_name, model in models.items():

                y_pred = model.predict(X)
                predictions[model_name] = y_pred

                y_prob = model.predict_proba(X)

                try:
                    auc = roc_auc_score(
                        y_true,
                        y_prob,
                        labels=model.classes_,
                        multi_class="ovr",
                        average="weighted"
                    )
                except ValueError:
                    auc = np.nan

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "AUC": auc,
                    "Precision": precision_score(
                        y_true, y_pred, average="weighted",
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_true, y_pred, average="weighted",
                        zero_division=0
                    ),
                    "F1": f1_score(
                        y_true, y_pred, average="weighted",
                        zero_division=0
                    ),
                    "MCC": matthews_corrcoef(y_true, y_pred)
                })

            # Display comparison
            st.header("Model Performance Comparison")

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df.style.format({
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "MCC": "{:.4f}"
                }),
                use_container_width=True
            )

            # Select model for detailed evaluation
            st.header("Detailed Model Evaluation")

            selected_model = st.selectbox(
                "Select a classifier",
                list(models.keys())
            )

            selected_predictions = predictions[selected_model]

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots(figsize=(9, 7))

            ConfusionMatrixDisplay.from_predictions(
                y_true,
                selected_predictions,
                ax=ax,
                xticks_rotation=45,
                cmap="Blues"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                selected_predictions,
                zero_division=0
            )

            st.text(report)
