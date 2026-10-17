
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    f1_score
)

# 1. Load dataset
data = pd.read_csv("loan_approval.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# 2. Separate input and output
X = data.drop(["Loan_ID", "Loan_Status"], axis=1)
Y = data["Loan_Status"]

# 3. Split data into training and testing
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# 4. Split training data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full,
    Y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=Y_train_full
)

# 5. Select numerical columns
num_colm = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

# 6. Select categorical columns
cat_colm = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area"
]

# 7. Numerical preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 8. Categorical preprocessing
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 9. Combine preprocessing
preprocessor = ColumnTransformer([
    ("nums", num_pipeline, num_colm),
    ("cats", cat_pipeline, cat_colm)
])

# 10. Create the complete model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# 11. Choose parameters to test
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [None, "balanced"]
}

# 12. GridSearchCV with 5-fold cross-validation
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

# 13. Train and find the best model
grid.fit(X_train, Y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)
print("Best CV Macro F1:", grid.best_score_)

# 14. Get validation probabilities
val_probabilities = best_model.predict_proba(X_val)[:, 1]

# 15. Select the best threshold using validation data
best_threshold = 0.5
best_val_f1 = -1

for threshold in [0.5, 0.6, 0.7, 0.8]:

    val_predictions = np.where(
        val_probabilities >= threshold,
        "Y",
        "N"
    )

    val_f1 = f1_score(
        Y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Threshold:", threshold,
        "Validation Macro F1:", val_f1
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_threshold = threshold

print("\nSelected Threshold:", best_threshold)

# 16. Final predictions on test data
test_probabilities = best_model.predict_proba(X_test)[:, 1]

Y_pred = np.where(
    test_probabilities >= best_threshold,
    "Y",
    "N"
)

# 17. Evaluate final model
print("\nFinal Test Accuracy:",
      accuracy_score(Y_test, Y_pred) * 100, "%")

print("\nConfusion Matrix (Y, N):")
print(confusion_matrix(
    Y_test,
    Y_pred,
    labels=["Y", "N"]
))

print("\nClassification Report:")
print(classification_report(
    Y_test,
    Y_pred,
    labels=["Y", "N"],
    zero_division=0
))

print("Y Recall:", recall_score(
    Y_test,
    Y_pred,
    pos_label="Y",
    zero_division=0
))

print("N Recall:", recall_score(
    Y_test,
    Y_pred,
    pos_label="N",
    zero_division=0
))

print("Final Macro F1:", f1_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0
))

# New applicant details

new_person = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}])

# Predict loan status
probability = best_model.predict_proba(new_person)[0]

prediction = np.where(
    probability[1] >= best_threshold,
    "Y",
    "N"
)

if prediction == "Y":
    print("Predicted Loan Status: Approved (Y)")
else:
    print("Predicted Loan Status: Not Approved (N)")

print("Approval Probability:", probability[1] * 100, "%")
