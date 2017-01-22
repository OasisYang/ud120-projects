#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################
from sklearn.metrics import accuracy_score

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# Use Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "The accuracy of Naive Bayes is " +str(acc)


## Use SVM
from sklearn import svm
c =100000
clf = svm.SVC(kernel='rbf',gamma='auto',C=c)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "The accuracy of SVM is " + str(acc)

## Use Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=10)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "The accuracy of Decision Tree is " + str(acc)

## Use KNN 

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "The accuracy of KNN is " + str(acc)

## Use AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME",
                         n_estimators=50)

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "The accuracy of AdaBoost is " + str(acc)

#Use Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(min_samples_split=20)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

print "The accuracy of Random Forest is " +str(acc)







try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
