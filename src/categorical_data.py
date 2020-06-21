"""
Three Approaches
Approach 1: Drop Categorical Variables
    The easiest approach to dealing with categorical variables is to simply remove them from the dataset. This approach will only work well if the columns did not contain useful information.

Approach 2:  Label Encoding
    Label encoding assigns each unique value to a different integer
    Works only with ordinal variables

Approach 3: One-Hot Encoding
    - One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data
    - In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data (e.g., "Red" is neither more nor less than "Yellow"). We refer to categorical variables without an intrinsic ranking as nominal variables.
    - One-hot encoding generally does not perform well if the categorical variable takes on a large number of values (i.e., you generally won't use it for variables taking more than 15 different values).
    - Issue: Need to handle the values that appear in the validation data but not in the training data

Reference: https://www.kaggle.com/alexisbcook/categorical-variables

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class HandleCategoricalData:
    def __init__(self):
        {}
    
    """    
    Drop columns with categorical values
    """
    def drop_columns(self, X_train, X_valid):
        # get names of columns with categorical values(have dtype as object)
        object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

        # drop columns in training and validation data
        X_train_without_cat = X_train.drop(object_cols, axis=1)
        X_valid_without_cat = X_valid.drop(object_cols, axis=1)

        return X_train_without_cat, X_valid_without_cat
    
    """
    Validate categorical values across train, validation and test set
    If there are different categorical values in training, validation and test set, our training algorithm will throw an error
    Solution1: make sure to have same categories across all dataset
    Solution2: only use the columns which contains same categorical values across all dataset and drop other categorical data columns
    This function takes input of X_train and X_valid dataset and compares categorical value across in each column and
    return the good column(having same categorical values) and bad columns(having different categorical values)
    """
    def good_bad_categorical_col(self, X_train, X_valid):
        # All categorical columns
        object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
        # Columns that can be safely label encoded
        good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]
        # Problematic columns that will be dropped from the dataset
        bad_label_cols = list(set(object_cols)-set(good_label_cols))  
        return good_label_cols, bad_label_cols
    
    """
    Label encode the categorical(ordinal) data.
    Note that it works bets only with ordinal data
    We loop over the categorical variables and apply the label encoder separately to each column.
    Output: Label encoded datasets
    """
    def label_encoding(self, categorical_cols, X_train, X_valid):
        label_X_train = X_train.copy()
        label_X_valid = X_valid.copy()

        # Apply label encoder to each column with categorical data
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            label_X_train[col] = label_encoder.fit_transform(X_train[col])
            label_X_valid[col] = label_encoder.transform(X_valid[col])

        return label_X_train, label_X_valid

    """
    One hot endode the categorical(nominal) data
    Important Parameters:
        - set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data,
        - setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
    Output: One hot encoded datasets
    """
    def one_hot_encoder(self, categorical_cols, X_train, X_valid, handle_unknown='ignore', sparse=False):
        # Apply one-hot encoder to each column with categorical data
        OH_encoder = OneHotEncoder(handle_unknown= handle_unknown , sparse= sparse)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_cols]))

        # One-hot encoding removed index; put it back
        OH_cols_train.index = X_train.index
        OH_cols_valid.index = X_valid.index

        # Remove categorical columns (will replace with one-hot encoding)
        num_X_train = X_train.drop(categorical_cols, axis=1)
        num_X_valid = X_valid.drop(categorical_cols, axis=1)

        # Add one-hot encoded columns to numerical features
        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

        return OH_X_train, OH_X_valid

if __name__ == '__main__':
    # kaggle dataset link: https://www.kaggle.com/c/home-data-for-ml-course/data?select=train.csv     
    _data_file_path = 'input/handle_missing_values_train.csv'
    
    # Read the data
    X_full = pd.read_csv(_data_file_path)
    print('Shape of the X_full data = ', X_full.shape) 

    ## Test method drop the rows which contain null values in target column
    ## Test method drop_columns()

    ## Test method good_bad_categorical_col()

    ## Test method label_encoding()

    ## Test method one_hot_encoder()
