"""
How to handle missing values.

One of the most important question you can ask yourself to help figure this out is this:

Is this value missing because it wasn't recorded or becuase it dosn't exist?

If a value is missing becuase it doens't exist (like the height of the oldest child of someone who doesn't have any children) then it doesn't make sense to try and guess what it might be. These values you probalby do want to keep as NaN. On the other hand, if a value is missing becuase it wasn't recorded, then you can try to guess what it might have been based on the other values in that column and row.

1. Delete Column/Row: 
- Here, we either delete a particular row if it has a null value for a particular feature and a particular column if it has more than 70-75% of missing values. 
- But make sure that column/row do not contain the important data
- Drawbacks: Suppose we have income feature in data. and its common for people with low or high income to not provide hsi info.

2. Imputation:
    - Mean: Use only when there are no outliers in data
    - Median: Can be used when there are outliers in the data
    - Drawbacks: - If affects the relationship between features. e.g. there are two columns age and income 
                   and if we replace the missing income value with mean then we may end with data like 10 year old having income of 1lac a month
                 - Cant use for categorical data    

3. Hypothesis: Study the other data and try to find correlation between availabel data and missing value.

4. Using ML itself to get missing data
    - Using KNN: Find K nearest rows and everage their value. It work with numerical data only
    - Deep Learning: Works great for categorical data but it takes more time and effort
    - Regression: use regression to find linear or non linear relation between missing feature and other values
    - MICE: Multiple Imputation by Chained Equations is most advance technique

Reference:
    - https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class HandleMissingData:
    def __init__(self):
        {}

    """
    Drop columns with missing values
    """
    def drop_columns(self, X_train, X_valid):
        # get names of columns with missing values
        cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

        # drop columns in training and validation data
        X_train_without_na = X_train.drop(cols_with_missing, axis=1)
        X_valid_without_na = X_valid.drop(cols_with_missing, axis=1)

        return X_train_without_na, X_valid_without_na

    """
    we use SimpleImputer to replace missing values with the mean value along each column.
    Although it's simple, filling in the mean value generally performs quite well (but this varies by dataset).
    While statisticians have experimented with more complex ways to determine imputed values (such as regression imputation, for instance),
    the complex strategies typically give no additional benefit once you plug the results into sophisticated machine learning models.
    
    strategy: string, default= 'mean'
        The imputation strategy.
        If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
        If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
        If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data.
        If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
    """
    def imputation(self, X_train, X_valid, strategy= 'mean'):
        my_imputer = SimpleImputer(strategy = strategy)
        X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
        X_valid_imputed = pd.DataFrame(my_imputer.transform(X_valid))
        # Imputation removed column names; put them back
        X_train_imputed.columns = X_train.columns
        X_valid_imputed.columns = X_valid.columns
        return X_train_imputed, X_valid_imputed

    """
    we use SimpleImputer to replace missing values with the mean value along each column.
    We impute the missing values, while also keeping track of which values were imputed
    
    strategy: string, default= 'mean'
        The imputation strategy.
        If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
        If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
        If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data.
        If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
    """
    def imputation_track(self, X_train, X_valid, strategy= 'mean'):
        # get names of columns with missing values
        cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

        # Make new columns indicating what will be imputed
        for col in cols_with_missing:
            X_train[col + '_was_missing'] = X_train[col].isnull()
            X_valid[col + '_was_missing'] = X_valid[col].isnull()

        my_imputer = SimpleImputer(strategy= strategy)
        X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
        X_valid_imputed = pd.DataFrame(my_imputer.transform(X_valid))
        # Imputation removed column names; put them back
        X_train_imputed.columns = X_train.columns
        X_valid_imputed.columns = X_valid.columns
        return X_train_imputed, X_valid_imputed


if __name__ == '__main__':
    # kaggle dataset link: https://www.kaggle.com/c/home-data-for-ml-course/data?select=train.csv     
    _data_file_path = 'input/handle_missing_values_train.csv'
    
    # Read the data
    X_full = pd.read_csv(_data_file_path)
    print('Shape of the X_full data = ', X_full.shape)  

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    print('Removed the rows with missing target values') 
    # To keep things simple, we'll use only numerical predictors
    X = X_full.select_dtypes(exclude=['object'])

    print('Keeping only numeric columns and removed all other columns') 
    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
    # Shape of training data (num_rows, num_columns)
    print('Shape of the training data = ', X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print('missing_val_count_by_column= \n', missing_val_count_by_column[missing_val_count_by_column > 0])

    hmd = HandleMissingData()    

    X_train_without_na, X_valid_without_na = hmd.drop_columns(X_train, X_valid)
    # Number of missing values after droping na columns
    missing_val_count_by_column = (X_train_without_na.isnull().sum())
    print('missing_val_count_by_column= \n', missing_val_count_by_column[missing_val_count_by_column > 0])

    # Similarly you can test imputation and imputation_track methods
