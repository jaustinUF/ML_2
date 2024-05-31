## Second machine learning project, finished 5/30/2024.
### Overview
- Project: train and test linear regression model to predict house price from features in a census dataset.
- Project dataset: housing.csv
- Original dataset: Dataset: "California House Price" https://www.kaggle.com/datasets/shibumohapatra/house-price
### Description
The project folder contains the dataset (housing.csv), and three Python script files:
- lin_reg_CA_ prep.py: data preparation 
- lin_reg_CA_ house_prices.py: training and testing the statsmodels OLS model
- lin_reg_CA_ with_sklearn.py: training and testing the sklearn LinearRegression model
### Details
#### Data preparation (lin_reg_CA_ prep.py)
After an initial view of the data and the data set information:
- missing data analysis: drop rows, use Imputation, or a more sophisticated ‘fix’
- dropped rows with missing data
- remove outliers with IQR method (visualized before/after with histogram)
- drop highly correlated feature (analyzed using heatmap
- converted categorical feature to ‘dummy’ binary features, using pandas . get_dummies
	-  merge dummy features with rest of data and drop ne dummy to eliminate "perfect multicolinearity"
#### OLS model training (lin_reg_CA_ house_prices.py)
text
#### Scikit-learn (sklearn) model training (lin_reg_CA_ with_sklearn.py)
text


Based on "Machine Learning for Beginners 2024" https://www.youtube.com/watch?v=43Bbjwy2f5I



