# DS.v2.5.3.3.5

# Spaceship Titanic Model

## Introduction

**Context**

Welcome to the year 2912. Humanity has reached the stars, and lost passengers need help to recover in a mysterious interdimensional incident.

The Spaceship Titanic, an interstellar vessel carrying over 13,000 emigrants to habitable exoplanets, suffered a tragic event near Alpha Centauri. A collision with a hidden spacetime anomaly caused nearly half of the passengers to be transported to an alternate dimension.

Fortunately, the ship’s damaged systems retained partial data. The mission is to use this data to build a predictive model that can identify which passengers were transported. This work will guide rescue efforts and potentially save thousands of lives.


**Goals:**

To build a binary classification model that predicts whether a passenger was Transported (True or False) using the features available in the dataset. Predictions will be evaluated based on classification accuracy on the test set.  

**Objectives:**


        Explore, clean, and preprocess the dataset.

        Train several machine learning models.

        Use techniques such as cross-validation, hyperparameter tuning, and ensembling to boost performance.

        Submit predictions to Kaggle following the required format.

        Achieve an accuracy score of at least 0.79 on the leaderboard.


**Dataset Overview**

The dataset contains information about 8,693 passengers included in the data set. It includes 14 independent variables (6 numerical, 7 categorical) and 1 target variable (Transported):

**Feature descriptions:**

PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.

HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.

CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.

Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.

Destination - The planet the passenger will be debarking to.

Age - The age of the passenger.

VIP - Whether the passenger has paid for special VIP service during the voyage.

RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.

Name - The first and last names of the passenger.

Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

### Requirements for Jupyter Notebook:

- Python: 3.11.9
- pandas: 2.2.3
- numpy: 1.26.4
- matplotlib: 3.10.1
- seaborn: 0.13.2
- scikit-learn: 1.6.1
- scipy: 1.11.4
- statsmodels: 0.14.4
- xgboost: 3.0.1
- lightgbm: 4.6.0
- catboost: 1.2.8
- optuna: 4.3.0
- shap: 0.47.2
- imblearn: 0.13.0
- phik: 0.12.4
- joblib: 1.4.2
- flaml: 2.3.5

## Data Source

The dataset is sourced from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic).

Jupyter Notebook and dataset source, clone the Repository:
[GitHub](https://github.com/TuringCollegeSubmissions/fverko-DS.v2.5.3.3.5)


## Jupyter Notebook Structure:

## 1. Introduction

## 2. Exploratory Data Analysis (EDA)

### A. Data loading & Initial checks

### B. Univariate Analysis

### C. Multivariate Analysis

## 3. Statistical Inference

## 4. Machine Learning Modeling

### A. Data Preparation and Feature Engineering

### B. Pipeline Preprocessing 

### C. Model Selection 

### D. Hyperparameter Tuning

### E. Ensembling

### F. FlaMl - Using AutoMl tools

## 5. Conclusion

This project involved a thorough investigation into predicting passenger transportation on the Spaceship Titanic, utilizing various machine learning models from traditional methods to advanced boosting algorithms and an ensemble approach. The initial exploration revealed LightGBM as the top performer among untuned models. Subsequent hyperparameter tuning significantly boosted both LightGBM and XGBoost performance on the training data.

A calibrated ensemble model was constructed, combining the strengths of the tuned LightGBM and XGBoost models. While the ensemble demonstrated a strong performance on the test set, even slightly outperforming tuned LightGBM in some metrics, its performance on the Kaggle competition (0.78910) surprisingly fell short of a simple tuned LightGBM model (0.80009). An automated machine learning (AutoML) solution, FlaML, emerged as the overall winner in the Kaggle competition (0.80476), highlighting the potential benefits of automated model selection and hyperparameter optimization.

Feature importance analysis, using both SHAP values and global importance metrics, consistently identified LuxurySpending, CryoSleep, and EssentialSpending as the most critical factors influencing a passenger's likelihood of being transported
