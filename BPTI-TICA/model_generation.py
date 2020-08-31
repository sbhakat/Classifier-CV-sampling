import numpy as np
from msmbuilder.utils import load
import os

plot_feat = load("./raw_features.pkl")
train_feat = load("./features.pkl")

df = load("./feature_descriptor.pkl")

from sklearn.linear_model import PassiveAggressiveClassifier

#Perception based model generation

X=np.vstack(plot_feat)
train_X=np.vstack(train_feat)

train_Y=np.concatenate([np.zeros(len(plot_feat[0])),
            np.ones(len(plot_feat[0]))])
if not os.path.isfile("./pasag_model_bpti.pkl"):
    train =True 
else:
    clf = load("./pasag_model_bpti.pkl")
    train =False
if train:
    clf = PassiveAggressiveClassifier(max_iter=1000)
    clf.fit(train_X, train_Y)

#Dumping the Model

if train:
    from msmbuilder.utils import dump
    dump(clf,"./pasag_model_bpti.pkl")

#Checking the clf

clf

#Printing out the Coefficients

coeff = ",".join([str(i) for i in clf.coef_[0]])
coeff

#Normalizing

w_norm = np.linalg.norm(clf.coef_)
func="(x+%s)/%s"%(str(clf.intercept_[0]),str(w_norm))
func
