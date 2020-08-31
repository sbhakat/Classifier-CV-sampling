# Classifier-CV-sampling
Binary classifier as CV in enhanced sampling

The general idea of binary classifier is to seperate probability densities corresponds to two (can be multiple) different states. However, how to seperate two states from a plethorea of data produced by MD simulation is not an easy task.

![classifier-idea](/classifier-general-idea.png)

There are several algorithms which can be used for this purpose. please have a look at https://scikit-learn.org/stable/supervised_learning.html#supervised-learning and choose your favourite classifier. Here are few classifiers which I personally tested

![classifier-idea](/classifier-algorithms.png)

One can use TICAs (see git https://github.com/sbhakat/BPTI-TICA-waterdynamics) as torch to seperate different states from MD simulation dataset and one can further use classifiers on those as depicted in the following picture. This requires less data.

![classifier-idea](/tica-classifier.png)

Problem set 1:

The first problem is a simple one. This is related to https://www.biorxiv.org/content/10.1101/2020.04.27.062539v1 
The flipping of tyrosine in Plasmepsin-II and BACE-1 leads to conformational changes. Chi1 angle of Tyr plays a critical role in this process. In plm-classifier notebook we will see how one can use binary classifier to seperate two states of Tyrosine and generate Plumed inputs which can be used as CVs to perform metadynamics.
