# crop-prediction
A machine learning model to predict crop types based on environmental and soil parameters
# Project Overview
This project aims to predict crop types based on environmental and soil parameters using machine learning. The dataset includes features like nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, rainfall, and crop labels. The goal is to classify 22 unique crop types with high accuracy. The final model achieves 99.5% accuracy using an optimized Random Forest Classifier.
# Dataset Details
The dataset contains 2,200 samples with the following features:

N, P, K: Soil nutrients (nitrogen, phosphorus, potassium).

Temperature: Environmental temperature.

Humidity: Environmental humidity.

pH: Soil pH level.

Rainfall: Annual rainfall.

Label: Crop type (22 unique crops).

Each crop has 100 samples, making the dataset balanced.
# Steps to Reproduce the Results
## 1-Data Preprocessing:

Load the dataset using Pandas.

Encode crop labels into numeric values using LabelEncoder.

Standardize features using StandardScaler.

Split the data into training (80%) and testing (20%) sets.

## 2-Logistic Regression:

Train a Logistic Regression model.

Evaluate the model using accuracy and classification report.

## 3-Random Forest Classifier:

Train a Random Forest model.

Evaluate the model using accuracy, classification report, and confusion matrix.

Visualize feature importance.

## 4-Hyperparameter Tuning:

Use Grid Search to optimize Random Forest hyperparameters.

Train the optimized model and evaluate its performance.

# Installation and Usage Instructions
## Installation
Clone the repository:

git clone https://github.com/your-username/crop-prediction.git
cd crop-prediction
## Install the required libraries:

pip install -r requirements.txt
Usage
## Open the Jupyter Notebook:

jupyter notebook notebooks/crop_prediction.ipynb
Execute each cell step-by-step to:

Load and preprocess the dataset.

Train and evaluate models.

Visualize results.

 # Future Work
Explore other models like XGBoost or Neural Networks.

Incorporate additional features such as soil type and seasonality.

Deploy the model as a web application for real-time predictions.
