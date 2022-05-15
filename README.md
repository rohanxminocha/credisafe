# CrediSafe
A Credit Card Fraud Detection System written in Python using Machine Learing by implementing KNN and K-means.

# Introduction
It's critical for credit card firms to be able to spot fraudulent credit card transactions so that customers aren't charged for things they didn't buy.

# Content
The dataset in [creditcard.csv](https://drive.google.com/file/d/1N4bflrjMX2FTWbCkQ2BFhMzvKDwcWDw0/view?usp=sharing) contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. <br><i>(Kaggle, https://www.kaggle.com/mlg-ulb/creditcardfraud)</i>

# Project Explaination
As the name suggests this repository is an classifies credit card transactions as fraudulent or genuine. It implements Machine Learning by making use of two alogorithms: k-Nearest Neighbours and k-means clustering.

## A few important aspects of the code:

### Data Cleaning
The Data Cleaning part of the system requires finding the attributes that need to be handled based on the data visualization task. It helps to ensure that the data is correct, consistent and usable. Functions should be developed as part of the data cleaning pipeline, and the dataset should be ready for further analysis.

### Data Visualization
The Data Visualization aspect of the data analysis needs to visualize the dataset and understand the distribution of the dataset.

### Confusion Matrix
The code also makes use of a confusion matrix to describe the performance of the classification model on a the data from creditcard.csv for which the true values are known.
