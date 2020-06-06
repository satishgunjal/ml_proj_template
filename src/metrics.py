from sklearn import metrics as skmetrics

#NOTE: As of now metrics for binary classification is implemented

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            'accuracy': self._accuracy,
            'f1': self._f1,
            'precision': self._precision,
            'recall': self._recall,
            'auc': self._auc,
            'logloss': self._logloss
        }

    def __call__(self, metric, y_true, y_pred, y_proba= None):
        if metric not in self.metrics:
            raise Exception('Mtric not implemented')
        if metric  == 'auc':
            if y_proba is not None:
                return self._auc(y_true= y_true, y_pred= y_proba)
            else:
                raise Exception('y_proba can not be None for AUC')
        if metric  == 'logloss':
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception('y_proba can not be None for logloss')
        else:
            return self.metrics[metric](y_true=y_true, y_pred = y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
        
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

if __name__ == '__main__':
    _y_true = [0,1,1,0,1,1]
    _y_pred = [0,1,0,0,1,1]    

    cm = ClassificationMetrics()('accuracy', y_true= _y_true, y_pred= _y_pred, y_proba = [0.5,0.5,0.5,0.5,0.5,0.6])
    print(cm)


