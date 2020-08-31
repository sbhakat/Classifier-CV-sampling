import numpy as np
import mdtraj as md
import pandas as pd
from msmbuilder.utils import load
from sklearn.linear_model import SGDClassifier

plot_feat = load("./raw_features.pkl")
train_feat = load("./features.pkl")

df = load("./feature_descriptor.pkl")

import os

X=np.vstack(plot_feat)
train_X=np.vstack(train_feat)

train_Y=np.concatenate([np.zeros(len(plot_feat[0])),
            np.ones(len(plot_feat[0]))])
if not os.path.isfile("./sgd_model_2.pkl"):
    train =True 
else:
    clf = load("./sgd_model_2.pkl")
    train =False
if train:
    clf = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
    clf.fit(train_X, train_Y)

if train:
    from msmbuilder.utils import dump
    dump(clf,"./sgd_model_2.pkl")


clf.coef_
