# NOTE! See lin_reg_CA_ prep.py for tests and comments on data preparation!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                         # visualization
import seaborn as sns                                   # visualization
# identify and interpret features (independent variables) that have a statistically significant impact on response
import statsmodels.api as sm                            # for causal analysis 2:16:40
from sklearn.model_selection import train_test_split

file_path = "housing.csv"
data = pd.read_csv(file_path)
## Missing Data Analysis
# Remove rows with missing values
data_cleaned = data.dropna()            # see dropna doco

pd.set_option('display.width', None)    # No wrapping to next line
pd.set_option('display.max_columns', None) # display all columns
## Data Exploration and Visualization 2:39:04
def histogram(df, col, title):
    """Show histogram of dataframe column with title"""
    sns.set(style = 'whitegrid')      # https://www.codecademy.com/article/seaborn-design-i
    plt.figure(figsize=(10, 6))             # figure size in inches
    sns.histplot(df[col], color='forestgreen', kde=True).set_title(label=title) # https://seaborn.pydata.org/generated/seaborn.histplot.html
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

## Inter-Quantile-Range [IQR] for removing outliers(2:50:36)
def remove_IQR_outliers(df_in, col):
    Q1 = df_in[col].quantile(0.25)
    Q3 = df_in[col].quantile(0.75)
    IQR = Q3 - Q1
    # print(f'median_house_value: first quantile = {Q1}   third quantile = {Q3}   Difference = {IQR}')
    # Bounds of wanted data (outliers removed)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # print(f"lower bound = {lower_bound}   upper bound = {upper_bound}")
    return df_in[
        (df_in[col] >= lower_bound) & (df_in[col] <= upper_bound)]

data_no_outliers_1 = remove_IQR_outliers(data_cleaned, 'median_house_value')

## BoxPlot for Outlier Detection and removal # (2:57:10)
def box_plot(df, col, title):
    sns.set(style='whitegrid')
    plt.figure(num=1, figsize=(8, 4.8))
    sns.boxplot(x=df[col], color='lightcyan')
    plt.title(title)
    plt.xlabel('Medial Income')
    plt.show()

data_no_outliers_2 = remove_IQR_outliers(data_no_outliers_1, 'median_income')
data = data_no_outliers_2
data = data.drop('total_bedrooms', axis = 1)
# note: 'dummies' columns datatype is 'bool' by default ... causes error in OLT fit below
ocean_prox_dum = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity', dtype=int)   # dtype forced to int
data = pd.concat([data.drop("ocean_proximity", axis =1), ocean_prox_dum], axis=1)
data = data.drop("ocean_proximity_ISLAND", axis = 1)
## Data preparation done!

## Split the Data into Train/Test (3:17:33)
features = list(data.columns)               # all variables names
features.remove('median_house_value')       # remove dependent variable name from features
target = ['median_house_value']             # 'target' is dependent variable name
x = data[features]                          # feature data
y = data[target]                            # dependent data
# Split the data into a training set and a testing set using train_test_split from sklearn
# split original df 80%/20% into train df/test df
# random_state: reandom number to control shuffling, for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)

## Training
# add 'const' column - so (this library's) fit calculates the intercept (b value in ax + b
#   https://stackoverflow.com/questions/41404817/statsmodels-add-constant-for-ols-intercept-what-is-this-actually-doing
x_train_const = sm.add_constant(x_train)    # Add a column of ones to array (3:22:55 - 3:24:15)
# fit (3:26:19)
model_fitted = sm.OLS(y_train, x_train_const).fit()
# print(model_fitted.summary())

## Prediction/Testing (3:47:50)
# Adding a constant to the test predictors
x_test_const = sm.add_constant(x_test)
# Making predictions on the test set
test_predictions = model_fitted.predict(x_test_const)
# print(test_predictions)

## Checking OLS Assumptions (3:49:27)
# Assumtion 1: Linearity
'''plt.scatter(y_test, test_predictions, color = "forestgreen")
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Observed vs Predicted Values on Test Data')
plt.plot(y_test, y_test, color='darkred')  # line for perfect prediction (true values)
plt.show()'''
# Assumption 2: Random Sample
# Calculate the mean of the residuals
mean_residuals = np.mean(model_fitted.resid)
print(f"The mean of the residuals is {np.round(mean_residuals,2)}")
# Plotting the residuals
'''plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color = "forestgreen")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()'''
# Assumption 3: Exogeneity
# Calculate the residuals
residuals = model_fitted.resid
# Check for correlation between residuals and each predictor
for column in x_train.columns:
    corr_coefficient = np.corrcoef(x_train[column], residuals)[0, 1]
    print(f'Correlation between residuals and {column}: {np.round(corr_coefficient,2)}')

## Assumption 4: Homoskedasticty (error term has constant variance) (3:57:39)
# same test as Assumption 2, but looking at shape rather than balance around zero

# See lin_reg_CA_ with_sklearn.py: redo of training in scikit-learn library;
#   suggests sklearn is better suited to machine learning
#   implication that methods above are more useful in causal/interpretive analysis
