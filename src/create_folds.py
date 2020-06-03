import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1

    # Code to shuffle the dataframe and reset the index
    #   The frac keyword argument specifies the fraction of rows to return in the random sample, so frac=1 means return all rows (in random order)
    #   specifying 'drop=True' prevents '.reset_index' from creating a column containing the old index entries.
    df = df.sample(frac = 1).reset_index(drop = True)
    print(df.head())

    # Lets split the training data into 5 batches(folds) each fold containing train and validation dataset
    #   Note here, with shuffle = False, random_state = 42 value don't affect the ordering of indices
    #   But it is required to keep the output same across multiple function call
    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle = False, random_state = 42)

    # Lets split the data and assign the fold number to each batch
    #   enumerate function will iterate through every value returned by split()
    #   kf.split() will split the data into training and validation set
    #   fold values will be 0, 1, 2, 3, 4
    #   remember we have added 'kfold' column in our dataset. We are going to assign the fold number to each new fold
    for fold , (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target.values)):
        print('kfold= %s, size of training data = %s, size of validation data = %s' % (fold, len(train_idx), len(val_idx)))
        df.loc[val_idx, 'kfold'] = fold

    # Using 'index = False' to prevent creating an index column
    df.to_csv('input/train_folds.csv', index = False)
    print('Training dataset splitted into 5 kfolds, new file train_folds.csv created in input folder')
