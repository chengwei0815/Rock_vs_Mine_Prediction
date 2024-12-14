# Rock vs Mine Prediction Using Logistic Regression

## Overview
This project uses a machine learning model to classify sonar data and predict whether an object is a rock or a mine. The model will be trained using a logistic regression algorithm, which is effective for binary classification problems.

## Dataset
The dataset consists of sonar data collected from a laboratory setup. The goal is to predict whether an object, based on its sonar features, is a rock or a mine.

- **Features:** 60 sonar readings (columns 0-59).
- **Target:** The last column, labeled either "R" (Rock) or "M" (Mine).

## Steps for Prediction

### 1. Data Collection and Processing
- We use a CSV file (`sonar_data.csv`) containing the sonar data.
- The dataset is loaded into a Pandas DataFrame using the `pd.read_csv()` function.
- The data is split into training and testing sets using `train_test_split`.

### 2. Model Training
- The logistic regression model is used to train the data.
- The model learns to classify objects based on sonar features (60 columns).
- The dataset is split into training data (90 data points) and testing data (10-20 data points).

### 3. Model Evaluation
- The model's accuracy is measured using the `accuracy_score()` function to check how well it predicts whether an object is a rock or mine.

## Workflow
1. **Import Libraries:**
   - `numpy`: Used for creating arrays.
   - `pandas`: Used for loading and processing data into DataFrames.
   - `train_test_split`: Splits the data into training and testing sets.
   - `LogisticRegression`: The machine learning model for binary classification.
   - `accuracy_score`: To calculate model accuracy.

2. **Load Dataset:**
   - Upload the `sonar_data.csv` file into Google Colab.
   - Use Pandas to load the CSV data into a DataFrame.
   - Use `.head()` to inspect the first five rows.

3. **Split Data:**
   - Split the dataset into features (60 columns) and the target (the last column, either "R" or "M").
   - Use `train_test_split` to create training and testing datasets.

4. **Train Logistic Regression Model:**
   - Train the model using the training dataset.

5. **Evaluate Model:**
   - Use the `accuracy_score()` function to check how well the model performs on the test dataset.

## Code Example

```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
sonar_data = pd.read_csv('sonar_data.csv', header=None)

# Split data into features and target
X = sonar_data.drop(60, axis=1)  # Features
y = sonar_data[60]               # Target (Rock or Mine)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100}%')
