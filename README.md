# EVALUATION OF STUDENT SATISFACTION WITH CAMPUS FACILITIES USING LOGISTICS REGRESSION
## Overview
This project focuses on evaluating student satisfaction with campus facilities using logistic regression. The study explores three key areas—lecture halls, laboratories, and libraries—assessing their adequacy and quality from the student perspective.

Data was collected through a survey using closed-ended questionnaires, and the responses were processed and analyzed using Excel and Python. The project applies descriptive statistics to summarize the data and logistic regression to model and predict the factors influencing student satisfaction levels.

The objectivesz if the study were as follows;
* To evaluate students’ satisfaction levels with key campus facilities.
* To analyze the adequacy and quality of lecture halls, laboratories, and libraries.
* To determine which factors significantly influence overall student satisfaction.
* To develop a predictive model using logistic regression to classify satisfaction outcomes.

#### Data Description
The dataset was collected through a structured questionnaire distributed among students.
Each response captured several key variables:
* Facility Type: Lecture Hall, Laboratory, or Library
* Adequacy Rating: Rated on a scale of 1 to 5, where 1 = Inadequate and 5 = Highly Adequate
* Quality Rating: Rated on a scale of 1 to 5, where 1 = Poor Quality and 5 = Excellent Quality
* Overall Satisfaction: Binary outcome variable indicating whether a student was Satisfied or Not Satisfied
* Program Level(Certificate, Diploma, Degree, Postgraduate)
* Frequency of use(daily, weekly, rarely)
#### Tools & Technologies
* Python
* pandas, numpy – Data processing
* matplotlib, seaborn – Visualization
* scikit-learn – Logistic regression and evaluation
### Load neccessary libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
```
### Load Data
Load the data from an Excel file into a Pandas DataFrame for analysis:
```python
data=pd.read_csv("Survey_data - Sheet1 (1).csv")
data.head(5)
data.info()
data.describe()
```
### Data Cleaning and Preparation
* The first step was to identify any missing data within the dataset.
```python
print(data.isnull().sum())
```
* Since the target variable Overall_satisfaction is critical for modeling, any rows missing this value were removed.
```python
data = data.dropna(subset=['Overall_satisfaction'])
```
* Categorical columns with  missing values were identified and replaced with the most frequent value of each column.
```python
# Select categorical columns
categorical_cols = data.select_dtypes(exclude=['number']).columns

# Fill missing values with mode
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
```
* Numeric columns with missing values were identified and replaced with the mean of each respective column.
```python
# Select numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns

# Fill missing values with mean
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
```
* Verifying if the data is clean
```python
data.info()
```
### Explaratory Data analysis(EDA)
***Pie chart*** - we used a piechart to visually represent how many students were satisfied versus not satisfied with the campus facilities.
```python
# pie chart of Overall_satisfaction
data['Overall_satisfaction'].value_counts().plot.pie(
    autopct='%1.1f%%', 
    startangle=90, 
    shadow=True, 
    figsize=(6,6),
    title='Overall_satisfaction Distribution'
)

plt.show()
```
***Bar chart*** - a bar chart was created to display the distribution of responses for the Frequency of Use variable.
It helped to highlight and understand frequency at which students used this facilities
```python
Barchart explaining frequency of use
data["Frequency_of_use"].value_counts().plot(kind="bar",color="skyblue")
plt.title("Distribution of Frequency of use")
plt.xlabel("Frequency of use")
plt.ylabel("count")
plt.show()
```
***Histogram*** - A histogram was used to visualize the distribution of students across different program levels(certificate, diploma, degree, post graduate)
This helps in understanding how the sample is spread out
```python
# Histogram of the distribution of program levels
plt.hist(data["Program_level"])
plt.title("Distribution of Program Level")
plt.xlabel("Program Level")
plt.ylabel("Frequency")
plt.show()
```
***Correlation heatmap*** - a correlation heatmap was created to explore the relationships between numerical variables in the dataset
```python
#finding correlation
correlation = data.select_dtypes("number").drop(columns="Overall_satisfaction").corr()
correlation
sns.heatmap(correlation)
```
### Split the Matrix
Split the data into Feature matrix and target matrix
```python
#Split Data
target= "Overall_satisfaction"
X= data.drop(columns=[target, "Program_level", "Frequency_of_use"])
y= data[target]
```
#### Train-test split
The dataset was divided into training and testing subsets before training the logistic regression model
```python
# Randomized train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check the shape of the resulting datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
```
#### Make a baseline model
Make a baseline model
```python
#baseline accuracy
acc_baseline=y_train.value_counts(normalize=True).max()
print("baseline Accuracy:", round(acc_baseline,2))
```
### Build Model
A logistic regression model was developed to predict student satisfaction based on key facility attributes such as adequacy, quality.

Logistic regression was selected because it is well-suited for binary classification problems determining whether a student is Satisfied or Not Satisfied with campus facilities.
```python
#build model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)

#fitting the model
model.fit( X_train,y_train)
```
### Model Evaluation
Accuracy Score measures how well the model correctly predicts whether students are Satisfied or Not Satisfied based on the test dataset.
```python
#accuracy score
accuracy_score(y_train,model.predict(X_train))
model.score(X_test,y_test)
cc_train=accuracy_score(y_train,model.predict(X_train))
acc_test=model.score(X_test,y_test)
print("Training Accuracy:", round(acc_train,2))
print("Test Accuracy:",round(acc_test,2))
````
### Prediction & Probability Estimates
Probability estimates were extracted to show the model’s confidence in each prediction.
Logistic regression naturally provides probability scores, making it useful for interpreting how strongly each input is associated with being Satisfied or Not Satisfied.
```python
#Predict Probability
model.predict(X_train)[:5]
y_train_pred_proba=model.predict_proba(X_train)
print(y_train_pred_proba[:5])
```
### Model interpretation
#### Feature Importance Analysis
Interpret how each feature (such as facility adequacy, quality) influenced student satisfaction.
Understanding these relationships helps identify which aspects of campus facilities most impact overall satisfaction.
```python
# Extract encoded feature names
features = model.named_steps['onehotencoder'].get_feature_names_out()
features
# Get logistic regression coefficients
importances = model.named_steps['logisticregression'].coef_[0]
importances
```
### Odds Ratio Analysis
Convert logistic regression coefficients into odds ratios for easier understanding.
While coefficients show the direction of the relationship (positive or negative), odds ratios reveal the magnitude of the effect each feature has on student satisfaction.
```python
# odds ratio
odds_ratios = pd.Series(np.exp(importances), index=features).sort_values()
print(odds_ratios)
```
#### Interpretation of Odds Ratio
* Odds Ratio > 1: The feature increases the likelihood of satisfaction.
* Odds Ratio < 1: The feature decreases the likelihood of satisfaction.
* Odds Ratio = 1: The feature has no effect on satisfaction.
### Visualization of Odds Ratios
Horizontal bar chart was plotted to visualize which features most strongly influence student satisfaction.
This visual representation helps quickly identify the most impactful and least impactful facility factors.
horizontal bar chart was plotted to visualize which features most strongly influence student satisfaction.
This visual representation helps quickly identify the most impactful and least impactful facility factors.
```python
#plot the odds ratio
odds_ratios.plot(kind="barh")
plt.xlabel=("odds Ratio")
```
## Project Interpretation
This project evaluated student satisfaction with campus facilities at Mount Kenya University, Thika, focusing on libraries, lecture halls, and laboratories.

The results showed that 74.3% of students were satisfied with the university’s facilities - a strong indication of overall positive perceptions. However, satisfaction varied across facilities: lecture halls performed best, while laboratories received lower ratings, highlighting areas that need attention.

The logistic regression model performed well (Accuracy = 81%) and revealed that facility quality mattered more than adequacy. In particular, laboratory quality (OR = 2.22) and library quality (OR = 2.01) were the strongest predictors of satisfaction — meaning students who rated these highly were over twice as likely to be satisfied.

Interestingly, satisfaction also varied by program level, with certificate students reporting lower satisfaction compared to degree students.

The findings show that improving the quality of learning spaces — especially labs and libraries can significantly enhance student satisfaction. The project demonstrates how logistic regression can turn survey data into clear, actionable insights to guide campus development and improve the overall student experience.



# Student Satisfaction Dashboard

This dashboard provides an overview of student satisfaction metrics across different program levels and facility usage.

## Dashboard Overview

![Distribution of Program Levels](distribution%20of%20program%20levels.png)
![Correlation Heatmap](correlation%20heatmap.png)
![Odds Ratio Analysis](horizontal%20bargraph%20of%20odds%20ratio.png)
![Overall Satisfaction](piei%20chart.png)
![Frequency of Use](distribution%20of%20frequency%20of%20use.png)

## Chart Descriptions

- **Program Level Distribution**: Shows the frequency distribution across different academic programs (Degree, Diploma, Certificate, Post Graduate)
- **Correlation Heatmap**: Displays correlations between various facility quality metrics (Library, Lab, Lecture Hall adequacy and quality)
- **Odds Ratio Analysis**: Horizontal bar graph showing the odds ratios for different facility quality factors
- **Overall Satisfaction**: Pie chart displaying the distribution of overall satisfaction levels (74.3% satisfied vs 25.7% not satisfied)
- **Frequency of Use**: Bar chart showing how frequently facilities are used by students (Daily, Weekly, Monthly, Rarely)

## Key Insights

- Majority of students (74.3%) report overall satisfaction
- Laboratory and library quality show the highest odds ratios in satisfaction analysis
- Degree programs appear to be the most common program level
- Daily usage appears to be the most frequent pattern of facility utilization


