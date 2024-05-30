import pandas as pd
import matplotlib.pyplot as plt                         # visualization
import seaborn as sns                                   # visualization

file_path = "housing.csv"
data = pd.read_csv(file_path)
# print(data.columns)
# print(data.head(10))
# print(data.info())
# print(data["ocean_proximity"].unique())

## Missing Data Analysis
'''
missing_values = data.isnull().sum()                        # .isnull works column-wise
missing_percentage = (missing_values / len(data)) * 100
print("Missing Values in Each Column:\n", missing_values)
print("\nPercentage of Missing Data:\n", missing_percentage)
'''
# Remove rows with missing values
data_cleaned = data.dropna()            # see dropna doco
# Verify that missing values have been removed
'''
print("\nMissing values in each column after removal:")
print(data_cleaned.isnull().sum())
'''
pd.set_option('display.width', None)    # No wrapping to next line
pd.set_option('display.max_columns', None) # display all columns
## Data Exploration and Visualization 2:39:04
#   describe  data through statistics and data visualization
# print(data.describe())                # basic statistics on each column
# note that lat/long identify areas so the data describes a BLOCK of houses, in that area.
# histogram of house values
def histogram(df, col, title):
    """Show histogram of dataframe column with title"""
    sns.set(style = 'whitegrid')      # https://www.codecademy.com/article/seaborn-design-i
    plt.figure(figsize=(10, 6))             # figure size in inches
    sns.histplot(df[col], color='forestgreen', kde=True).set_title(label=title) # https://seaborn.pydata.org/generated/seaborn.histplot.html
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
# show 'data_cleaned' histogram
df_name = 'data_cleaned'
col_name = 'median_house_value'
# histogram(data_cleaned, col_name, f"Histogram of '{col_name}' from '{df_name}'")

## Inter-Quantile-Range [IQR] for removing outliers(2:50:36)
#       see lin_reg_CA_ to_IQR_2-57-10.py for look at weirdness with IQR and 5000001 bin
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
# print(data_no_outliers_1.shape)

## BoxPlot for Outlier Detection and removal # (2:57:10)
'''sns.set(style = 'whitegrid')
plt.figure(num = 1, figsize=(8,4.8))
sns.boxplot(x=data_no_outliers_1['median_income'], color='lightcyan')
plt.title('Outlier Analysis of Median Income')
plt.xlabel('Medial Income')
plt.show()'''

def box_plot(df, col, title):
    sns.set(style='whitegrid')
    plt.figure(num=1, figsize=(8, 4.8))
    sns.boxplot(x=df[col], color='lightcyan')
    plt.title(title)
    plt.xlabel('Medial Income')
    plt.show()

data_no_outliers_2 = remove_IQR_outliers(data_no_outliers_1, 'median_income')
data = data_no_outliers_2
# box_plot(data, 'median_income', 'Outlier Analysis of Median Income')
'''plt.figure(figsize=(8, 8*(8/12)))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='Greens', annot_kws={"fontsize": 8})
plt.title('Correlation Heatmap of Housing Data')
plt.show()'''
# total_rooms, total_bedrooms, population, households show high correlation with each other.
#   thus the dark center squares in the heatmap.
#->  (3:09:49) notes one of the four high-correlation has low correlation with the dependent variable (value)
#       so chooses to drop it ('total_bedrooms')
#       note did not choose to drop the lowest correlation, 'population'.
data = data.drop('total_bedrooms', axis = 1)
'''plt.figure(figsize=(8, 8*(8/12)))               # heatmap with 'total_bedrooms' dropped
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='Greens', annot_kws={"fontsize": 8})
plt.title('Correlation Heatmap of Housing Data')
plt.show()'''

# deal with categorical data (ex. strings in 'ocean_proximity')
# "String Data Categorization to Dummy Variables"   # (3:10:37)
print(data["ocean_proximity"].unique())             # tutorial uses for loop incase multiple categorical data columns
# (3:11:43) convert categorical column data to binary (True/False) 'dummy' columns
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
ocean_prox_dum = pd.get_dummies(data['ocean_proximity'], prefix='ocean_proximity', dtype=int)   # 'dtype=int needed to keep values numeric
print(ocean_prox_dum.info())
print(ocean_prox_dum.head())
'''# https://pandas.pydata.org/docs/reference/api/pandas.concat.html
#   apparently the object parameter allows a sequence ... could have dropped 'ocean_proximity' separately
data = pd.concat([data.drop("ocean_proximity", axis =1), ocean_prox_dum], axis=1)
# drop one of the dummies to eliminate "perfect multicolinearity" (3:14:49)
data = data.drop("ocean_proximity_ISLAND", axis = 1)
print(data.columns)
'''





