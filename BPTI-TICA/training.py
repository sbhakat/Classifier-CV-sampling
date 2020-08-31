from msmbuilder.featurizer import DihedralFeaturizer
import numpy as np
import mdtraj as md
import pandas as pd
from msmbuilder.utils import load,dump
import os

a = np.array([166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
       179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
       192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
       205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
       218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
       231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
       244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
       257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
       270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282,
       283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295,
       296, 297, 298, 299, 300, 301, 302, 303, 304, 534, 535, 536, 537, 
       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 
       551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 
       564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 
       577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589,    
       590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 
       603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 
       616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628])


trj_list = [md.load_dcd("/home/sbhakat/Aurora/DESRES/Conv_trajectory/DCD_files/Conv-BPTI-all-0000.dcd",top="prot_maeconv.pdb", atom_indices = a), md.load_dcd("/home/sbhakat/Aurora/DESRES/Conv_trajectory/DCD_files/Conv-BPTI-all-0001.dcd",top="prot_maeconv.pdb", atom_indices = a), md.load_dcd("/home/sbhakat/Aurora/DESRES/Conv_trajectory/DCD_files/Conv-BPTI-all-4068.dcd",top="prot_maeconv.pdb", atom_indices = a), md.load_dcd("/home/sbhakat/Aurora/DESRES/Conv_trajectory/DCD_files/Conv-BPTI-all-4069.dcd",top="prot_maeconv.pdb", atom_indices = a)]

dump(trj_list, "traj_list.pkl")


f=DihedralFeaturizer(types=['phi','psi'])
dump(f,"raw_featurizer.pkl")

feat = f.fit_transform(trj_list)
dump(feat, "raw_features.pkl")


f=DihedralFeaturizer(types=['phi','psi'])
dump(f,"featurizer.pkl")

df1 = pd.DataFrame(f.describe_features(trj_list[0]))
dump(df1,"feature_descriptor.pkl")

feat = f.fit_transform(trj_list)
dump(feat, "features.pkl")
