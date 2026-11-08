# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification algorithms to predict the variety of dry beans based on their physical measurements.

Five classification algorithms are implemented and compared using standard evaluation metrics. The trained models are also deployed through a Streamlit web application, allowing users to upload test data and evaluate the performance of the classifiers without retraining them.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

The Dry Bean Dataset contains 13,611 instances and 16 numerical input features extracted from images of dry beans. Each instance belongs to one of seven dry bean varieties.

**Target variable:** `Class`

**Bean varieties:**
- SEKER
- BARBUNYA
- BOMBAY
- CALI
- HOROZ
- SIRA
- DERMASON

**Input features (16):**
1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

The dataset was divided into training and testing sets using an 80:20 split with stratification to preserve the class distribution.

Feature scaling was applied within the pipelines for Logistic Regression, KNN, and SVM.

## 3. GitHub Repository Link

**GitHub Repository:** [(https://github.com/tanisha01data/BITS-ML-Assignment-2)]

The repository contains:
- `app.py` – Streamlit application.
- `requirements.txt` – Required Python dependencies.
- `README.md` – Project documentation.
- `ML_Assignment_2.ipynb` – Source code and model implementation.
- `test_data.csv` – Test dataset used for evaluation.
- `model_comparison.csv` – Model performance comparison.
- `models/` – Saved trained models and feature-name artifact.

## 4. Models Used and Evaluation

The following five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models were evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.9056 | 0.9619 | 0.9060 | 0.9056 | 0.9057 | 0.8858 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

### Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved an accuracy of 92.07% and an AUC of 0.9934. It performed consistently across the evaluation metrics, demonstrating strong classification performance.

**Decision Tree**

The Decision Tree classifier achieved an accuracy of 90.56%. It provided good classification performance but performed slightly below Logistic Regression and SVM across the reported metrics.

**K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 91.66%. It showed competitive performance, although its overall results were slightly lower than those of SVM and Logistic Regression.

**Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 76.39%, the lowest among the five models. Despite its relatively high AUC of 0.9644, its precision, recall, F1-score, and MCC were comparatively lower.

**Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 92.21%, an AUC of 0.9937, and an MCC of 0.9059. It demonstrated the strongest overall performance across the reported evaluation metrics.

### Overall Winner

**Support Vector Machine (SVM)** was the best-performing classifier on the selected test dataset. It achieved the highest values across all six reported evaluation metrics, making it the overall winner in this experiment.

## 5. Streamlit Application

The project is deployed using Streamlit Community Cloud.

**Live Application:** [(https://bits-ml-assignment-2-gygj9thhebt4psgkj7bul6.streamlit.app/)]

The application provides the following features:
- CSV upload option for test data.
- Model selection dropdown.
- Performance comparison of all five classifiers.
- Evaluation metrics including Accuracy, AUC, Precision, Recall, F1-score, and MCC.
- Confusion matrix and classification report for the selected model.

The application loads the previously saved trained models and evaluates the uploaded test dataset without retraining.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Jupyter Notebook

## 7. Conclusion

This project compares five machine learning classification algorithms on the Dry Bean Dataset. SVM achieved the best overall results, followed closely by Logistic Regression and KNN. Gaussian Naive Bayes showed comparatively lower classification accuracy.

The trained models were saved and integrated into a Streamlit application to enable interactive evaluation using uploaded test data.
