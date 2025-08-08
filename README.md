# 🚢 Titanic Survival Prediction - Kaggle Classification Challenge

This repository contains my solution for [Titanic - Machine Learning from Disaster – Kaggle](https://www.kaggle.com/competitions/titanic) competition.
The goal of this challenge is to predict the survival of passengers aboard the Titanic using passenger information such as age, sex, ticket class, and more.
It is a binary classification task with the target variable being Survived (0 = No, 1 = Yes).


![173261197316135550](https://github.com/user-attachments/assets/6997e7ef-430c-40ac-a1f1-b1d95ef61e99)


---

## 🗂️ Project Structure

```

project-root/
├── 📄 titanic_task.ipynb         # Main notebook: full ML pipeline from preprocessing to submission
├── 📊 train.csv                  # Training dataset with features and target ('Survived')
├── 🧪 test.csv                   # Test dataset: features only
├── 📝 gender_submission.csv      # Template for Kaggle submission
└── 🚀 submission1.csv            # Submission from LogisticRegression
└── 🚀 submission2.csv            # Submission from RandomForestClassifier
└── 📜 README.md                  # Project documentation

```

---

## 💻 Technologies Used

- **Language**: Python 3.x  
- **Environment**: Jupyter Notebook


## 📦 Libraries

- `pandas`, `numpy` – Data manipulation  
- `matplotlib`, `seaborn` – Visualization  
- `scikit-learn` – Preprocessing & modeling
- `LabelEncoder` – Categorical encoding 
- `GridSearchCV` – Hyperparameter tuning  
- `StandardScaler` – Feature scaling 
- `PolynomialFeatures` – Non-linear feature expansion
- `Pipeline` – Combine preprocessing and modeling steps 
- `LogisticRegression` – Binary classification model
- `train_test_split` - Train-validation split
- `RandomForestClassifier` – Ensemble-based classification model
- `warnings` – Suppress warnings

---

## 📊 Dataset Description

Competition: Competition: [Titanic – Machine Learning from Disaster – Kaggle](https://www.kaggle.com/competitions/titanic).

- `train.csv`: Passenger data with `survival` labels
- `test.csv`: Passenger data without labels (for prediction)
- `gender_submission.csv`: Required format for submitting predictions

---

## 🔁 Workflow Summary

The `titanic_task.ipynb` notebook implements the complete pipeline from data cleaning to model training and final submission.

### 1. 📥 Data Loading & Preprocessing
- Loaded `train.csv` and `test.csv`.
- Dropped unnecessary columns: `PassengerId`, `Name`, and `Cabin`.
- Cleaned the `Ticket` column by extracting numeric parts only.
- Handled missing values:
  - `Age`: filled via interpolation
  - `Fare`: filled with mode (in `test.csv`)
  - `Embarked`: filled with mode (in `train.csv`)

### 2. 🧪 Feature Encoding & Scaling

- Encoded all categorical features using `LabelEncoder`
- Built a preprocessing pipeline using:
  - `PolynomialFeatures` (degree=2) – for non-linear feature expansion
  - `StandardScaler` – for feature scaling
  - `LogisticRegression` – used as the first (baseline) classification model  
  - `RandomForestClassifier` – used as the second (ensemble-based) model  

### 3. 🧠 Model Training & Hyperparameter Tuning

- Combined preprocessing and modeling steps using `Pipeline`
- Tuned model using `GridSearchCV` with:
  - Penalty types: `l1`, `l2`, `elasticnet`, `none`
  - Solvers: `liblinear`, `saga`, `lbfgs`, etc.
  - Parameters: `max_iter`, `l1_ratio` (when applicable)
- Split training data into training and validation sets (`test_size=0.2`, `random_state=41`)


### 4. 🧠 Prediction & Submission

- Trained final model on the training data
- Evaluated on the validation set
- Generated predictions on test data



### 5. 📤 Prediction & Submission

Final predictions for each model were saved:

- `submission1.csv` → LogisticRegression
- `submission2.csv` → RandomForestClassifier

---

## 📈 Results Summary

| Model                  | Accuracy |    Output File    |
|------------------------|----------|-------------------|
| LogisticRegression     | 0.84     | `submission1.csv` |
| RandomForestClassifier | 0.87     | `submission2.csv` |

---

## ⚙️ Installation

To install all required libraries:


```bash
pip install pandas numpy matplotlib seaborn scikit-learn
