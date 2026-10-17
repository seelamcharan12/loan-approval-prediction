# Loan Approval Prediction

Hi! This is a machine learning project I built to predict whether a loan application may be approved or not, based on details such as income, credit history, education, and property area.

I created this project to practise the machine learning workflow from start to finish — preparing the data, training a model, checking its performance, and trying predictions for new applicants.

## What does it do?

The model predicts one of two outcomes:

- `Y` — Loan approved
- `N` — Loan not approved

It uses the `Loan_Status` column as the target. The `Loan_ID` column is removed because it is only an identifier.

## How I built it

Here is the process I followed:

1. Loaded the dataset using Pandas.
2. Split the data into training, validation, and test sets.
3. Filled missing numerical values using the median and missing categorical values using the most common value.
4. Scaled numerical data and converted categorical data into a format the model can use.
5. Built a pipeline with preprocessing and Logistic Regression.
6. Used `GridSearchCV` with 5-fold cross-validation to compare model settings.
7. Compared a few prediction thresholds using validation data.
8. Checked the final model using test data.

## Tools and libraries

- Python
- Pandas
- NumPy
- scikit-learn
- Logistic Regression
- GridSearchCV and cross-validation

## Results

These are the results from my current run:

| Metric | Result |
|---|---:|
| Test accuracy | 86.18% |
| Macro F1 score | 0.815 |
| Recall for approved applications (`Y`) | 0.988 |
| Recall for not-approved applications (`N`) | 0.579 |
| Selected threshold | 0.5 |
| Best cross-validation Macro F1 | 0.710 |

### Confusion matrix

The rows are the actual labels and the columns are the model's predictions. The label order is `Y`, then `N`.

| Actual / Predicted | `Y` | `N` |
|---|---:|---:|
| `Y` | 84 | 1 |
| `N` | 16 | 22 |

The model identified approved applications well in this run. It did not identify all applications that were actually marked `N`, so improving that part is one area I would like to explore further.

## How to run the project

Make sure Python is installed, then keep these files in the same folder:

```text
loan-approval-project/
├── code.py
├── loan_approval.csv
└── README.md
```

Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

Run the program:

```bash
python code.py
```

On Windows, you can also try:

```bash
py code.py
```

The script prints the selected model settings, validation threshold scores, and final test results. The new-applicant prediction code can also be used after the model has been trained.

## What I learned

While working on this project, I practised handling missing data, encoding categorical features, scaling numerical features, building pipelines, using cross-validation, tuning model settings, and evaluating predictions with more than just accuracy.

## Future improvements

- Try other classification algorithms and compare their results.
- Improve how well the model detects `N` applications.
- Add a simple interface to enter applicant details.
- Save the trained model so it can be reused without training every time.
- Add the dataset source and its licence details.

## Note

This is a learning project. The model's output is only a prediction based on the dataset and does not guarantee whether a bank will approve a real loan.

## Author

**Charan**  
Add your GitHub profile link here.
