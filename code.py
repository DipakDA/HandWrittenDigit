from PIL import Image
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

train = pd.read_csv('train.csv')
test_features = pd.read_csv('test.csv')
train_features = train.drop('label', 1)
train_target = train['label']



svc = svm.LinearSVC()
svc.fit(train_features, train_target)
y_pred = svc.predict(test_features)

pickle.dump(y_pred, open("save.p", "wb"))
