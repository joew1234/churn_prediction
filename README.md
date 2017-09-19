# Churn Prediction: Ride Sharing Services

## Overview
Predicted customers churning for a ride sharing app, defined as not having used the app for 30 days. Final model was logistic regression, which gave us an estimated 4% increase in profits compared to our baseline model.

## Dataset
The dataset was provided by Galvanize. It had 11 features including categorical, numerical and text.

## Feature Engineering and EDA
We found the most signal in splitting users into mostly weekend or weekday users, as well as splitting by phone type (iPhone, Android,...).

## Model Development
We wanted maximum interpretability from our model so that we could give more specific business recommendations. For this reason, we mostly iterated through logistic regression models with different features that we had engineered. We also used a random forest model as a benchmark for model performance. Finally, we built a pipeline to train, test and predict quickly on new data.

## Result and Inference
Our final logistic model performed comparably to the random forest. Our recommendation is to send coupons to those who the model predicts will churn, which we estimate will increase profits by 4% compared to doing nothing.

## Credits
This project would not have been possible without my teammates Rosina Norton, Edward Rha and D.A. Nguyen.
