# Project Name
This project creates a multiple linear regression model to predict daily demand for shared bikes in the U.S. market using the BoomBikes dataset. The goal is to understand how different factors—season, weather, temperature, humidity, holidays, etc.—affect total bike rentals (cnt) so the company can plan operations and improve revenue post-pandemic.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)


<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Objective
BoomBikes wants to understand:

 - Which variables significantly affect bike demand
 - How well these variables explain variations in rentals
 - How demand may change after the COVID-19 situation stabilizes

The model will guide strategic decisions like pricing, fleet allocation, marketing, and expansion.

- Data Preparation 
   Converted categorical variables (season, weathersit) into string labels
   Created dummy variables using pd.get_dummies(drop_first=True) to avoid multicollinearity
   Scaled numerical variables using StandardScaler
   Dropped unnecessary columns (instant, dteday, casual, registered)
   Ensured no missing or infinite values
   Split the data into train (70%) and test (30%)


## Technologies Used
- Python - version 3.9
- Pandas - version 2.0
- python - version 3.9.6
- seaborn - version 0.13.2
- matplotlib - version 3.9.4
- pandas - version 2.3.1
- sklearn - version 1.6.1


## Conclusions
The linear regression model successfully identifies the most important drivers of bike rental demand. Temperature, weather conditions, seasonality, and year-over-year growth significantly affect usage. The insights from this model can help BoomBikes plan operations, optimize supply, and strategize for post-pandemic recovery.
