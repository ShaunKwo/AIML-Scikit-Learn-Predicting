# AI and Machine Learning

## By Shaun Kwo Rui Yu

---

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Datasets](#datasets)
4. [Methodology](#methodology)
   - Exploratory Data Analysis
   - Classification
   - Regression
5. [Results](#results)
   - Classification Results
   - Regression Results
6. [Conclusion](#conclusion)
7. [Recommendations](#recommendations)
8. [References](#references)

---

## Introduction

This project involves using AI and machine learning techniques to analyze and predict water quality and hospital costs. The tasks include data preprocessing, feature analysis, model building, hyperparameter tuning, and model evaluation.

---

## Objectives

1. To perform exploratory data analysis on the given datasets.
2. To build machine learning models for predicting water quality based on various water properties.
3. To construct regression models to predict hospital costs.
4. To compare model performances using evaluation metrics.

---

## Datasets

- **Classification Dataset:** Used for predicting water quality. Contains features like `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, and `Clarity`.
- **Regression Dataset:** Used for predicting hospital costs.

---

## Methodology

### Exploratory Data Analysis

- **Data Overview:** Initial inspection of the dataset, handling missing values, and statistical analysis.
- **Feature Analysis and Data Visualization:** 
  - Heatmap to check feature correlations.
  - Histogram to observe frequency distributions.
  - Boxplots to identify anomalies.

### Classification

- **Data Overview:** Loading and preparing the dataset, handling missing values.
- **Feature Analysis:** Heatmap, histogram, and boxplots for feature analysis.
- **Modeling:** Building classification models to predict water quality.
- **Evaluation:** Using metrics like accuracy, confusion matrix, and F1 score to evaluate model performance.

<h5>Name : Shaun Kwo Rui Yu</h5>

<hr></hr>
<h1>Classification Dataset for Water Quality</h1>
<hr></hr>
<h5><b>The objectives of this water quality classification project are to:</b></h5>
<ol>
<li>Explore the provided dataset and gain insights into its structure and characteristics.</li>
<li>Build a classification model to predict water quality (0 or 1) based on various features.</li>
</ol>

<h5><b>Background Info:</b></h5>
The dataset contains information related to water quality, including features that could influence the classification of water quality. The goal is to develop a model that accurately predicts water quality to aid in environmental monitoring and decision-making.

<h5><b>Additional Info:</b></h5>
Water quality classification involves determining whether a water sample meets certain standards or criteria, indicating whether it is safe for various purposes such as drinking, recreation, or ecological health.

Potential Features Influencing Water Quality:
<ol>
<li>Chemical composition of the water (e.g., pH Levels, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes Concentration)</li>
</ol>

<h5><b>Steps in the Project:</b></h5>
<ol>
<li><b>Step 1: Exploratory Data Analysis (EDA)</b>
   <ol>
      <li>1.1 Data Overview: Load the dataset, check for missing values, duplicated entries, and basic statistics.</li>
      <li>1.2 Feature Analysis: Visualize feature distributions, explore correlations, and consider feature engineering.</li>
      <li>1.3 Target Variable Distribution: Understand the distribution of the target variable (Quality).</li>
   </ol>
</li>
<li><b>Step 2: Data Preprocessing</b>
   <ol>
      <li>2.1 Handling Missing Values: Identify and handle missing values.</li>
      <li>2.2 Feature Scaling and Normalization: Standardize numerical features and apply normalization if necessary.</li>
      <li>2.3 Encoding Categorical Variables: Encode categorical variables into numerical representations.</li>
   </ol>
</li>
<li><b>Step 3: Model Selection</b>
   <ol>
      <li>3.1 Define the Problem: Clearly define the problem as a binary classification task.</li>
      <li>3.2 Model Choices: Select candidate classification algorithms (Logistic Regression, Decision Trees, Random Forest, SVM, Gradient Boosting).</li>
      <li>3.3 Train-Test Split: Split the dataset into training and testing sets.</li>
   </ol>
</li>
<li><b>Step 4: Model Training and Evaluation</b>
   <ol>
      <li>4.1 Train the Models: Train each selected model on the training set.</li>
      <li>4.2 Evaluate Model Performance: Assess metrics such as Accuracy, Precision, Recall, and F1-score.</li>
      <li>4.3 Hyperparameter Tuning: Fine-tune hyperparameters for selected models.</li>
   </ol>
</li>
<li><b>Step 5: Dummy Baseline and Model Interpretability</b>
   <ol>
      <li>5.1 Dummy Baseline: Establish a dummy classifier for baseline comparison.</li>
   </ol>
</li>
<li><b>Conclusion</b>
   <ol>
      <li>Summarize key findings, insights, and challenges.</li>
      <li>Provide recommendations for further improvement or exploration.</li>
   </ol>
</li>
</ol>


### Regression

- **Modeling:** Building regression models to predict hospital costs.
- **Hyperparameter Tuning:** Fine-tuning model parameters to improve performance.
- **Evaluation:** Using metrics like RMSE and MAPE to evaluate model performance.

<h1><b>Regression for Predicting Hospitalisation Cost($)</b></h1>
<h5>Name : Shaun Kwo Rui Yu</h5>

<hr></hr>

<h5><b>The objectives of this assignment are to:</b></h5>
This dataset comprises information related to hospital patients and includes 1338 data points with 7 columns.

<ul>
  <li><b>ID:</b> Unique identifier for each patient</li>
  <li><b>Age:</b> Age of the patient</li>
  <li><b>Gender:</b> Gender of the patient</li>
  <li><b>BMI:</b> Body Mass Index of the patient</li>
  <li><b>Smoker:</b> Smoking status of the patient (yes/no)</li>
  <li><b>Region:</b> Region of residence of the patient</li>
  <li><b>Cost ($):</b> Hospital cost incurred by the patient</li>
</ul>

<h5><b>Steps in the Project:</b></h5>
<ol>
  <li><b>Step 1: Exploratory Data Analysis (EDA)</b>
    <ol>
      <li>1.1 Data Overview: Load the dataset, check for missing values, duplicated entries, and basic statistics.</li>
      <li>1.2 Feature Analysis: Visualize feature distributions, explore correlations, and consider feature engineering.</li>
      <li>1.3 Target Variable Distribution: Understand the distribution of the target variable (Cost).</li>
    </ol>
  </li>
  <li><b>Step 2: Data Preprocessing</b>
    <ol>
      <li>2.1 Handling Missing Values: Identify and handle missing values.</li>
      <li>2.2 Feature Scaling and Normalization: Standardize numerical features and apply normalization if necessary.</li>
      <li>2.3 Encoding Categorical Variables: Encode categorical variables into numerical representations.</li>
    </ol>
  </li>
  <li><b>Step 3: Model Selection</b>
    <ol>
      <li>3.1 Define the Problem: Clearly define the problem as a regression task.</li>
      <li>3.2 Model Choices: Select candidate regression algorithms (Linear Regression, Decision Trees, Random Forest, Gradient Boosting).</li>
      <li>3.3 Train-Test Split: Split the dataset into training and testing sets.</li>
    </ol>
  </li>
  <li><b>Step 4: Model Training and Evaluation</b>
    <ol>
      <li>4.1 Train the Models: Train each selected model on the training set.</li>
      <li>4.2 Evaluate Model Performance: Assess metrics such as Mean Absolute Error, Mean Squared Error, and R-squared.</li>
      <li>4.3 Hyperparameter Tuning: Fine-tune hyperparameters for selected models.</li>
    </ol>
  </li>
  <li><b>Step 5: Dummy Baseline and Model Interpretability</b>
    <ol>
      <li>5.1 Dummy Baseline: Establish a dummy regressor for baseline comparison.</li>
      <li>5.2 Model Interpretability: Analyze feature importance for interpretability.</li>
    </ol>
  </li>
  <li><b>Conclusion</b>
    <ol>
      <li>Summarize key findings, insights, and challenges.</li>
      <li>Provide recommendations for further improvement or exploration.</li>
    </ol>
  </li>
</ol>

---

## Results

### Classification Results

- **Task Definition:** Predicting water quality based on features.
- **Output Variable:** "Quality" (binary: 1 for clean, 0 for dirty).
- **Model Performance:** Detailed evaluation using accuracy and other metrics.

### Regression Results

- **Best Model:** Gradient Boosting with hyperparameter tuning.
- **Best Parameters:** `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}`
- **Model Performance:** Compared to a dummy baseline model, the tuned Gradient Boosting model showed significantly lower MAPE and RMSE values, indicating better performance.

---

## Conclusion

From this assignment, several key learnings were achieved:

- **Data Handling:** Effective preprocessing, handling missing values, encoding, and feature scaling.
- **Modeling and Tuning:** Experience with scikit-learn for model building and hyperparameter tuning.
- **Evaluation Skills:** Proficiency in assessing model quality using various metrics.
- **Analysis and Interpretability:** Insight into feature importance and model performance.

---

## Recommendations

- Continuously improve data preprocessing techniques.
- Experiment with more advanced models and tuning strategies.
- Utilize the insights from feature importance to refine models further.

---

