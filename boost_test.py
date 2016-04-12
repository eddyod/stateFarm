import numpy as np
import pandas as pd 
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import xgboost as xgb

DATA_PATH = "data/"


def load_data(datafile):
    """ use pandas to read the csv """
    print "loading ",datafile
    return  pd.read_csv(datafile, low_memory=True)

train = load_data(DATA_PATH + 'train_features.csv')

target = 'label'
skip = ['img_id', target]
features = [col for col in train.columns if col not in skip]
#y_train = train['label'].values
#Convert target labels to numbers
LE_target = preprocessing.LabelEncoder()
target_transform = LE_target.fit_transform(train.label)
y_train = target_transform

X_train = train[features].values
train = None
print "X_train",type(X_train),np.shape(X_train)

print "Splitting data"
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.25,random_state=42)

num_class = len(np.unique(y_train))
params = {"objective": "multi:softprob", "num_class": num_class, "eta": 0.05, "max_depth": 10, "seed": 42,"eval_metric": "mlogloss","silent":1}
early_stopping_rounds = 150
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
dtest = xgb.DMatrix(X_test, label=y_test, missing=np.NaN)
evallist = [(dtest,'validate'), (dtrain,'train')]
clf = xgb.train(params, dtrain, early_stopping_rounds, evallist)
