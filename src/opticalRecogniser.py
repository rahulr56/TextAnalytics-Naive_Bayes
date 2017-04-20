print "Setting up system"
print "Importing libraries"
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neural_network as sknn

from itertools import cycle
from sklearn.svm import SVC
from datetime import datetime as dt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import column_or_1d
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cvs
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

#seed=int(str(dt.now()).translate(None,":-. "))
i=dt.now()
seed=int(i.strftime("%H%M%S"))

## Read training Data file
print "Reading training dataset"
try:
    data = pd.read_csv(open('../DataSet/DigitRecognition/optdigits_raining.csv'))
except Exception as e:
    print "Error in opening Tarining Dataset"
    print "Error Message : "+str(e)
    exit(1)

## Read Test dataset
print "Reading test dataset"
try:
    testdata = pd.read_csv(open('..//DataSet/DigitRecognition/optdigits_test.csv'))
except Exception as e:
    print "Failed to open testing data set : "+str(e)
    ip=input("Do you wish to continue(Y/N)?").lower()
    if ip.contanins('y'):
        print "Proceeding without test data set..."
        pass
    else:
        print "Exitting after Cleanup..."
        data.close()
        exit(1)

for x in ['a1','a8','h1','h8']:
    del data[x]
    del testdata[x]


## Separate features and results in training dataset
print "Processing training dataset"
targetDataResult = (data['result'])[200:]#.reshape(-1,1)
targetFeatures = list(data.columns[:-1])
targetDataFeatures=data[targetFeatures][200:]
crossValidationTrain=data[targetFeatures][:200]
crossValidationTrainResults=data['result'][:200]


print "Processing test dataset"
## Separate features and results in Test dataset
testtarget = testdata['result']
testFeatureList = list(testdata.columns[:-1])
testResults=testdata['result']
testFeatures=testdata[testFeatureList]
## Fit the training data set into Multi Layer Perceptron classifier
## This builds us a Neural Network
print "Building a Naive Bayes Network from training data"
#svm_map = SVC(kernel='linear',cache_size=900)
#svm_map.fit(targetDataFeatures,targetDataResult)
#gnb=GaussianNB()
#gnb.fit(targetDataFeatures,targetDataResult)
#print "GNB Cross Validation score : "+str(cvs(gnb,X=crossValidationTrain, y=crossValidationTrainResults,cv=5))
#print "GNB Prediction rate is "+str(gnb.score(testFeatures,testResults)*100)

mnb=MultinomialNB(alpha=100.005)
mnb.fit(targetDataFeatures,targetDataResult)
print "MNB Cross Validation score : "+str(cvs(mnb,X=crossValidationTrain, y=crossValidationTrainResults,cv=5))
print "MNB Prediction rate is "+str(mnb.score(testFeatures,testResults)*100)
#bnb=BernoulliNB()
#bnb.fit(targetDataFeatures,targetDataResult)
#print "BNB Cross Validation score : "+str(cvs(bnb,X=crossValidationTrain, y=crossValidationTrainResults,cv=5))
#print "BNB Prediction rate is "+str(bnb.score(testFeatures,testResults)*100)
## Predict test data based on the Neural Network formed from training dataset
#predictedTestResults=list(knc_map.predict(testFeatures))

#actualTestResults=list(testResults)
#print "Prediction complete"
#print "Calculating Prediction results"
#s=f=0
#for i in range(len(actualTestResults)-2):
#	if actualTestResults[i] == predictedTestResults[i]:
#		s=s+1
#	else:
#		f=f+1
#print "--------------------------------------------------------------------------"
#print "\n\n"
#print "Percentage of data that is predicted correctly is "+str(float(s)/len(actualTestResults)*100.0)
#print "Percentage of data that is predicted wrong is "+str(float(f)/len(actualTestResults)*100.0)
#print "Prediction rate is "+str(knc_map.score(testFeatures,testResults)*100)
#print "Prediction rate is "+str(float(s)/len(actualTestResults)*100.0)
#print "\n"
#print "--------------------------------------------------------------------------"
