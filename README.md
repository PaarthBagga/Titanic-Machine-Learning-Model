# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project aims to predict passenger survival on the Titanic using machine learning. I used **Pandas for data manipulation**, **NumPy for numerical operations**, and **Scikit-learn for preprocessing and model training**. My process involved **cleaning the data, encoding categorical features, handling missing values, and training a Decision Tree and Random Forest model** to find the best approach.

---

## 🛠️ Libraries Used

I used the following Python libraries:
- **NumPy (`numpy`)** – For numerical computations.
- **Pandas (`pandas`)** – For reading, cleaning, and processing data.
- **Scikit-learn (`sklearn`)**:
  - `DecisionTreeClassifier` – To test tree-based model performance.
  - `RandomForestClassifier` – To build a more robust model.
  - `mean_absolute_error` – For evaluating model accuracy.
  - `train_test_split` – To split data into training and validation sets.
  - `OneHotEncoder` – To convert categorical data into numerical form.

---

## 📂 Dataset
I used the **Kaggle Titanic dataset**, which includes:
- `train.csv`: Passenger data including survival status.
- `test.csv`: Passenger data (for making predictions).
- `gender_submission.csv`: Sample submission file.

It contains the following features:
- **PassengerId**: Unique ID of the passenger
- **Survived**: Survival status (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1st, 2nd, or 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Passenger age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number (if available)
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 🔍 My Process

### 1️⃣ **Loading the Data**
I loaded the dataset and checked for missing values:


import pandas as pd

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

### 2️⃣ Handling Missing Values
- Embarked column: Filled missing values with "Unknown".
- Age & Fare: Replaced NaNs with the median value of their Pclass groups.

```
train_data['Embarked'] = train_data['Embarked'].fillna('Unknown')
train_data['Age'] = train_data['Age'].fillna(train_data.groupby('Pclass')['Age'].transform('median'))
test_data['Age'] = test_data['Age'].fillna(test_data.groupby('Pclass')['Age'].transform('median'))
test_data['Fare'] = test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform('median'))
```

### 3️⃣ Encoding Categorical Features
Since Sex and Embarked are categorical and important variables that determine survival, I applied One-Hot Encoding:
```
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
```
Encoding 'Sex' and 'Embarked' for training data
```
ohe_sex_train = ohe.fit_transform(train_data[["Sex"]])
ohe_embarked_train = ohe.fit_transform(train_data[["Embarked"]])
```
Encoding 'Sex' and 'Embarked' for test data
```
ohe_sex_test = ohe.fit_transform(test_data[["Sex"]])
ohe_embarked_test = ohe.fit_transform(test_data[["Embarked"]])
```
Merging encoded columns and dropping the original ones
```
train_data = pd.concat([train_data, ohe_sex_train, ohe_embarked_train], axis=1).drop(columns=["Sex", "Embarked"])
test_data = pd.concat([test_data, ohe_sex_test, ohe_embarked_test], axis=1).drop(columns=["Sex", "Embarked"])
```
### 4️⃣ Feature Selection
I selected features that are most relevant for survival prediction:
```
features = ["Pclass", "Sex_female", "Sex_male", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
X = train_data[features]
y = train_data["Survived"]
```
### 5️⃣ Splitting the Data
I divided the dataset into training and validation sets:
```
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
```
---

## 🌳 Decision Tree vs. Random Forest
I then tested which model worked better for the dataset,
### Decision Tree Classifier
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=25)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(val_y, predictions)
```
### Random Forest Classifier
```
from sklearn.ensemble import RandomForestClassifier
def get_mae_rf(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, random_state=25)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    return mean_absolute_error(val_y, predictions)
```
### Determining the Best Model
```
import numpy as np

best_tree = min(range(2, 102), key=lambda x: get_mae(x, train_X, val_X, train_y, val_y))
best_rf = min(range(2, 102), key=lambda x: get_mae_rf(x, train_X, val_X, train_y, val_y))

print(f"Best Decision Tree Leaf Nodes: {best_tree}")
print(f"Best Random Forest Leaf Nodes: {best_rf}")
```
### Based on the lowest MAE, I determined which model was the most accurate
```
rf_model = RandomForestClassifier(max_leaf_nodes=36, random_state=25)
rf_model.fit(X, y)
```
---

## 🚀 Generating Predictions
```
predictions = rf_model.predict(test_data[features])
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

This was the approach I took in the Titanic Machine Learning from Disaster competition!
