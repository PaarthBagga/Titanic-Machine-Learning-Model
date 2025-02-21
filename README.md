# Titanic Survival Prediction Project

## Overview
This project aims to analyze the Titanic dataset and build a machine learning model to predict passenger survival. The dataset includes features such as age, gender, class, and fare price, which are used to determine the likelihood of survival.

## Dataset
The dataset used for this project is the **Titanic - Machine Learning from Disaster** dataset from Kaggle. It contains the following features:

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

## Project Workflow
1. **Data Cleaning & Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature engineering (creating new relevant features)

2. **Exploratory Data Analysis (EDA)**
   - Visualizing survival distribution
   - Analyzing correlations between features

3. **Feature Selection & Engineering**
   - Selecting the most relevant features
   - Normalization and scaling (if necessary)

4. **Model Training**
   - Training multiple machine learning models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - Gradient Boosting (XGBoost)
   - Evaluating models using accuracy, precision, recall, and F1-score

5. **Model Evaluation & Interpretation**
   - Comparing model performance
   - Feature importance analysis

6. **Deployment (Optional)**
   - Creating a web app (Flask or Streamlit) for predictions

## Installation
To run the project, install the necessary dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
