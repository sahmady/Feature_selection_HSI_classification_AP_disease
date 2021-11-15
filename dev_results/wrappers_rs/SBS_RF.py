import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, learning_curve, ShuffleSplit, GridSearchCV
from time import process_time
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.utils import shuffle


# load dataset 10% of original dataset
X = np.load("dataset/X.npy")
y = np.load("dataset/y.npy")

# make 1% of dataset to be ready for further use
X = X[::10, :]
y = y[::10]

# shuffle dataset before training
X, y = shuffle(X, y, random_state=42)

columns = np.load("dataset/columns.npy")

clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)

# record starting time
tic = process_time()

sfs1 = sfs(clf, 
           k_features=1, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=5,
           n_jobs=-1)

sfs1 = sfs1.fit(X, y, custom_feature_names=columns)

# save the result of wrapper method
filename = 'results/wrappers/SBS_RF.sav'
pickle.dump(sfs1, open(filename, 'wb'))


# record ending time

toc = process_time()

runtime_duration = toc - tic

print(runtime_duration)