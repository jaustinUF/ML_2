# NOTE! See lin_reg_CA_ prep.py for tests and comments on data preparation!
#   see lin_reg_CA_ house_prices.py for training with statsmodels.api
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt                         # visualization
import seaborn as sns                                   # visualization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
# random_state: random number to control shuffling, for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)

## redo linear regression training/testing with scikit-learn library: https://scikit-learn.org/stable/ (3:58:13)
# more 'machine learning' focused ("traditional machine learning side")
# Scaling the Data
# initialize
scaler = StandardScaler()
# Fit to data, then transform ... scale training data & learn scaling parameters (3:59:59)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
# instantiate and fit model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
# Predict based on scaled test data
y_pred = lr.predict(x_test_scaled)
# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f'RMSE on Test Set: {rmse}')          # output metrics
#   error of $60K ... probably because some of the assumtions were 'violated'
#    (4:03:35) notes course intent is not pure/best ML fit, rather chance to show casual and interpretive analysis.
#   (4:03:59) next steps to improve data prep, model, and fit
