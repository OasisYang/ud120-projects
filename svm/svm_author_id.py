#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

#########################################################
t0 = time()
c = 10000.0
svc = svm.SVC(kernel='rbf', gamma='auto', C=c).fit(features_train, labels_train)
print "training time", round(time() - t0, 3),'s'

t1 = time()
pred = svc.predict(features_test)
print "predicting time", round(time() - t1, 3), 's'

# svc_rbf = svm.SVC(kernel='rbf', gamma='auto', C=c).fit(features_train, labels_train)
# svc_poly = svm.SVC(kernel='poly', degree=3, C=c).fit(features_train, labels_train)
# lin_svc = svm.LinearSVC(C=c).fit(features_train, labels_train)

print "svc with rbf kernel is " + str(svc.score(features_test, labels_test))
# print "svc with rbf kernel is " + str(svc.score(features_test, labels_test))
# print "svc with poly kernel is " + str(svc.score(features_test, labels_test))
# print "Linear svc is " + str(svc.score(features_test, labels_test)) 
print pred[10],pred[26],pred[50],sum(pred)
