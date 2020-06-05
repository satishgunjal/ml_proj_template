import pandas as pd
from sklearn import model_selection

class CrossValidation:
    def __init__(
        self,
        df,
        target_cols,
        shuffle,
        problem_type = 'binary_classification',
        multilabel_delimiter = ',',
        num_folds = 5,
        random_state = 42
        ):        
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.shuffle = shuffle
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.num_folds = num_folds
        self.random_state = random_state

        """
        Code to shuffle the dataframe and reset the index
        The frac keyword argument specifies the fraction of rows to return in the random sample, so frac=1 means return all rows (in random order)
        specifying 'drop=True' prevents '.reset_index' from creating a column containing the old index entries.
        """
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac= 1).reset_index(drop = True)
        """
        Add kfold column, later we will update it with actual value of fold for each record
        """
        self.dataframe['kfold'] = -1

    """
    Split the data in kfolds
    nunique()> To get the distinct values
    StratifiedKFold()> To keep the ratio of positive and negative label values same in each fold
    split()> Generates indices to split data into train and test sets
             enumerate function will iterate through every indices returned by split()
             fold values will be 0 to self.num_folds - 1
             remember we have added 'kfold' column in our dataset. We are going to assign the fold number to each new fold
    """
    def split(self):
        if self.problem_type in ('binary_classification', 'multiclass_classification'):
            if self.num_targets != 1:
                raise Exception('Invalid number of targets for this problem type')

            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception('Only one unique value found')
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits= self.num_folds, shuffle=False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X= self.dataframe, y= self.dataframe[target].values )):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type in ('single_col_regression', 'multi_col_regression'):
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception('Invalid number of targets for this problem type') 
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception('Invalid number of targets for this problem type')

            kf = model_selection.KFold(n_splits = self.num_folds, shuffle = False)
            # Only X value required
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        
        #Suitable for timeseries data and when data size is very large
        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split("_")[1])
            print('holdout_percentage= ', holdout_percentage)
            num_holdout_samples = int( len(self.dataframe) * holdout_percentage / 100)
            print('num_holdout_samples= ', num_holdout_samples)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == 'multilabel_classification':
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits= self.num_folds, shuffle=False)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X= self.dataframe, y= target)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception('Problem type not understood')
        
        return self.dataframe

if __name__ == '__main__':
    
    ######## Testing for problem_type = "binary_classification" ######## 
    #_data_file_path = 'input/train.csv'
    #_target_cols = ['target']
    #_problem_type = "binary_classification"
    #_shuffle = True

    ######## Testing for problem_type = "single_col_regression" ########      
    #_data_file_path = 'input/house_price_regression_train.csv'
    #_target_cols = ['SalePrice']
    #_problem_type = "single_col_regression"
    #_shuffle = True

    ######## Testing for problem_type = "holdeout" ########      
    #_data_file_path = 'input/train.csv'
    #_target_cols = ['target']  # Not required  TODO modify the class to handle it
    #_problem_type = "holdout_10"
    #_shuffle = False

    ######## Testing for problem_type = "multilabel_classification" ######## 
    # kaggle dataset link: https://www.kaggle.com/c/imet-2020-fgvc7/data?select=train.csv     
    _data_file_path = 'input/multilabel_classification_train.csv'
    _target_cols = ["attribute_ids"] 
    _problem_type = "multilabel_classification"
    _multilabel_delimiter = " "
    _shuffle = True

    df = pd.read_csv(_data_file_path)
    print('Shape of the data = ', df.shape)     
    cv = CrossValidation(df, shuffle=_shuffle, target_cols=_target_cols, problem_type=_problem_type, multilabel_delimiter= _multilabel_delimiter)    

    df_split = cv.split()
    print('Count of kfold values in each fold= \n ', df_split.kfold.value_counts())
    
    print('Top 5 rows from final datframe= \n ', df_split.head())

    if _problem_type in ('binary_classification', 'multiclass_classification'):
        print('% distribution of classes in given data= \n ', df[_target_cols[0]].value_counts(normalize =True))
        for fold in range(df_split.kfold.nunique()):
            print('% distribution of classes for kfold= ' + str(fold) + ' is \n', df_split[_target_cols[0]].value_counts(normalize =True))                



