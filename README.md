# Gstn_Hackathon-2024
My Team was able to achieve a **97.58%** accuracy on a predictive model using **Random Forest** training algorithm for the problem statement given for the [GST Hackathon](https://innovateindia.mygov.in/online-challenge-for-developing-a-predictive-model-in-gst/ "About gst hackathon")
<br>Please find the detailed solution in the python [file1](file:///C:/Users/Dell/Desktop/gstn%20hackathon-final.pdf "random forest accuracy"), [file2](file:///C:/Users/Dell/Desktop/Random_Forest.pdf "Random forest metrics"). Also, there is a python [Google Colab](https://colab.research.google.com/drive/1Dj2ZciIako1es8NtCRhEg2wrr6J_goL8 "Drive") which contains preliminary work done by My Team.<br>
# Get Started
This repository contains code and data analysis for the submission of **Analytics Hackathon on Developing a Predictive Model in GST** by **My team**.
<br>Please follow the below instructions to validate the results of the predictive model.
## Pre-requisites
#### Core Technologies 
1)Python: The primary programming language used in this project.<br>
2)Google colabs: For interactive development and documentation.<br><br>
**Data Processing and Analysis**
<br>1)pandas: For data manipulation and analysis.<br>
2)scikit-learn: For machine learning algorithms and tools.<br><br>
**Visualization**
<br>1)Matplotlib For creating static, animated, and interactive visualizations.<br>
2)PyQt6: For displaying matplotlib graphs in CLI.<br><br>
**Machine Learning Models**<br>
This project utilizes the following machine learning model:<br><br>
Random Forest: For both classification and regression tasks, providing ensemble learning capabilities.
<br><br>Please ensure you have the appropriate versions of these libraries installed before running the project. Refer to the requirements.txt file for specific version information.
## Model Construction
|Aspect|Logistic Regression|Random Forest|
|---|---|---|
|Model Type |<ul><li>Linear model for classification</li></ul><ul><li>Estimates probability of class membership</li></ul>|<ul><li>Ensemble of decision trees</li></ul><ul><li>Combines multiple trees for classification</li></ul>|
|Pros|<ul><li>Simple and interpretable</li></ul><ul><li>Efficient with large datasets</li></ul><ul><li>Performs well with linearly separable classes</li></ul><ul><li>Provides probability scores</li></ul><ul><li>Less prone to overfitting on small datasets</li></ul>|<ul><li>Handles non-linear relationships well</li></ul><ul><li> Robust toutliers and noise</li></ul><ul><li>Provides feature importance rankings</li></ul><ul><li> Reduces overfitting through ensemble learning</li></ul><ul><li>Can capture complex interactions between features</li></ul>|
|Cones|<ul><li>Assumes linear relationship between features and log-adds of the outcome</li></ul><ul><li> May underperform with complex, non-linear relationships</li></ul><ul><li>Sensitive to outliers</li></ul><ul><li>Limited ability to handle feature interactions without explicit engineering</li></ul>|<ul><li>Less interpretable than logistic regression</li></ul><ul><li>Can be computationally expensive for very large datasets</li></ul><ul><li>May overfit on small, noisy datasets if not tuned properly</li></ul><ul><li>Predictions are not as easily probabilistic as logistic regression</li></ul>|
|Use Cases|<ul><li>When interpretability is crucial (e.g.. healthcare, finance)</li></ul><ul><li>Large-scale linear problems</li></ul><ul><li>As a baseline model</li></ul>|*<ul><li> Complex datasets with non-linear relationships</li></ul><ul><li>When feature importance is needed</li></ul><ul><li>Problems where predictive accuracy is more important than interpretability</li></ul>|
## Methodology
Basically methodology means a set of guidelines, rules, and process that help project teams manage and complete projects.<br>
In this, The methodology we follow is...<br>
**Step 1:** **Data cleaning**<br>
In this step we clean the given gst datasets. Initially Datasets contains Nan values(Null), so we replace them with mean/average values and filtering irrelevant inforamation.<br> 
### Performance Comparison
The peformance of these models can vary depending on the specific characteristics of your datasets<br>
1. **Dataset Size**<br>
* Logistic Regression often performs well on large datasets.
* Random Forest can excel with medium sized datasets but might struggle with very large ones due to Computational costs.
