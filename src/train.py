import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

#from . import dispatcher

TRAINING_DATA = 'input/train_folds.csv' # os.environ.get('TRAINING_DATA')
FOLD = 0 # int(os.environ.get('FOLD'))
MODEL = 'randomforest' # os.environ.get('MODEL')

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]
    print('kfold= %s, shape of train_df training data = %s, shape of valid_df validation data = %s' % (FOLD, train_df.shape, valid_df.shape))

    # Lets create label data
    ytrain = train_df.target.values
    yvalid = valid_df.target.values
    print('kfold= %s, shape of ytrain label data = %s, shape of yvalid label data = %s' % (FOLD, ytrain.shape, yvalid.shape))

    #Lets drop the unwanted columns 
    train_df = train_df.drop(['id', 'target', 'kfold'], axis = 1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis = 1)
    print('kfold= %s, shape of train_df training data = %s, shape of valid_df validation data = %s' % (FOLD, train_df.shape, valid_df.shape))

    # To make order of variables same in validation an training datasets
    valid_df = valid_df[train_df.columns]
    
    """
    Convert categorical data inot numeric data using LabelEncoder
    LabelEncoder()>  Encode the labels with numeric value starting from 0
    Using '.values.tolist()' to get array object
    transform()> Transform labels to normalized encoding (numeric values)   and replace the existing values with it
    label_encoders list to store columns and LabelEncoder() instance
    """
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())             
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist()) 
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c, lbl))

    # Data is ready to train
    # Default value of n_estimators = 100    
    clf = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1, verbose = 2)
    #clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    predictions = clf.predict_proba(valid_df)[:, 1]
    print('Prediction = ', predictions)
    # Use roc_auc_score when data is skewed
    print('roc_auc_score = ', metrics.roc_auc_score(yvalid, predictions))
    

