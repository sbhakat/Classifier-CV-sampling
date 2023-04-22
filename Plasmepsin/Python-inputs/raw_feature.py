#msmbuilder imports 
from msmbuilder.dataset import dataset
from msmbuilder.featurizer import ContactFeaturizer
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import ContinuousTimeMSM
from msmbuilder.utils import verbosedump,verboseload
from msmbuilder.cluster import KCenters
from msmbuilder.utils import load,dump

#other imports
import os,glob,shutil
import numpy as np
import mdtraj as md
import pandas as pd
import pickle
#prettier plots

from msmbuilder.utils import load,dump

#Loading the trajectory
a = np.arange(1119,1277)
trj_list = [md.load("../../traj-210-300.xtc",top="../prot.pdb", atom_indices = a), md.load("../../traj-380-470.xtc",top="../prot.pdb", atom_indices = a)]
#dump(trj_list, "traj_list.pkl")

#from msmbuilder.utils import load,dump

f = DihedralFeaturizer(types=['chi1', 'chi2'])
dump(f,"raw_featurizer.pkl")


feat = f.transform(trj_list)

dump(feat, "raw_features.pkl")


f=DihedralFeaturizer(types=['chi1', 'chi2'])
dump(f,"featurizer.pkl")
df1 = pd.DataFrame(f.describe_features(trj_list[0]))
dump(df1,"feature_descriptor.pkl")
feat = f.transform(trj_list)

dump(feat, "features.pkl")
