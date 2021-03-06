print "Setting up the system..."
print "Importing Libraries"
import re
import csv
import string
import subprocess
import pandas as pd
from nltk import bigrams
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
from datetime import datetime as dt
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB
from nltk.probability import FreqDist as nF
from sklearn.utils.validation import column_or_1d
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import cross_val_score as cvs
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

##Global variables
stopWords=[]
dataSet=[]
decisionAttributes=[]
i=dt.now()
seed=int(i.strftime("%H%M%S"))

## Function that initalizes the system
def initializeSystem():
    print "Preparing set of stop words"
    stop = stopwords.words('english') + punctuation + ['rt', 'via','i\'m','us','it']
    for x in stop:
    	stopWords.append(stemmer.stem(lemmatiser.lemmatize(x, pos="v")))

## Function to convert string into a list of token represented in the regular expression
def tokenize(s):
    return tokens_re.findall(s)

## Converts the input string into a list of words
## Each word is lemmatized and stemmed after the stop words are removed
def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else stemmer.stem(lemmatiser.lemmatize(token.lower(), pos="v")) for token in tokens]
    return tokens

def processString(string):
    terms_stop = [term for term in preprocess(string) if term not in stopWords and len(str(term)) >1 and not term.isnumeric()]
    return terms_stop

## Reads the file and returns it as a dataset
def loadFile(filePath):
    print "Loading the dataset..."
    try:
        fileRead = open(filePath,"r")
    except Exception as e:
        print "Unable to open training or testing data"
        print "Error Message : "+str(e)
        print "Exitting..."
        exit(1)

    try:
        reader=csv.reader(fileRead,dialect='excel')
    except Exception as e:
        print "Error in reading file."
        print "Error Message: "+str(e)
        exit(1)

    for row in reader:
        temp=(row[1],row[-1])
        dataSet.append(temp)

## Generates a sparse matrix using the reviews
def prepareSparseMatrix(convertedReviews):
    sparseMatrix=[]
    for cr in convertedReviews:
        newCr=[0]*len(decisionAttributes)
        for word in cr:
            if word in decisionAttributes:
                index=decisionAttributes.index(word)
                newCr[index]+=1
            else:
                pass
        sparseMatrix.append(newCr)
    return sparseMatrix

## Removes stop words, stems and lemmatizes each word in all the reviews passed
def convertReviews(reviews):
    convertedReviews=[]
    for a in reviews:
        convertedReviews.append(processString(str(a).lower()))
    return convertedReviews

## Calculates the decision attributes and returns the most frequent ones.
def getDecisionAttributes(convertedReviews) :
    toCount=[]
    for a in convertedReviews:
        toCount.append(" ".join(a))
    str1=""
    for a in toCount:
        str1+="".join(a)
    x=Counter(str1.split(" "))
    for (k,v) in x.most_common(min(500,len(x))):
        decisionAttributes.append(k)


#########################################################
#                                                       #
#                 MAIN STARTS HERE                      #
#                                                       #
#########################################################

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    r'<[^>]+>', # HTML tags
    r"(?:[a-z][a-z\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

print "Defining grammar.."
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()


initializeSystem()


print "Analysing training dataset.."
loadFile("../DataSet/AmazonDataSet/part_amazon_baby_train.csv")
## Parse the training dataset read from the file
trainDataFeaturesReviews = pd.DataFrame(dataSet,columns=["review","rating"])
targetRating = (trainDataFeaturesReviews['rating']).reshape(-1,1)
targetReview = trainDataFeaturesReviews['review']

#crossValidationTrain=(trainDataFeaturesReviews['review'])[:300]
#crossValidationTrainResults=((trainDataFeaturesReviews['rating'])[:300]).reshape(-1,1)

print "Analysing testing dataset.."
loadFile("../DataSet/AmazonDataSet/amazon_baby_test.csv")
## Parse the testing dataset read from the file
testDataFeaturesReviews = pd.DataFrame(dataSet,columns=["review","rating"])
testReview = testDataFeaturesReviews['review']
testRating = testDataFeaturesReviews['rating']

## Convert the reviews by removing stop words, lemmatize and stem each word
print "Preprocessing the data set..."
trainReviews = convertReviews(targetReview)

## Find the decision attributes from the converted reviews
getDecisionAttributes(trainReviews)

## Build a sparse matrix using the reviews from training dataset
trainSparseMatrix = prepareSparseMatrix(trainReviews)

## Build a sparse matrix using the reviews from testing dataset
testSparseMatrix=prepareSparseMatrix(convertReviews(testReview))

## Build a Neural Network
dataFeatures = pd.DataFrame(trainSparseMatrix,columns=decisionAttributes)
testDataFeatures = pd.DataFrame(testSparseMatrix,columns=decisionAttributes)

## Predict the test data based on the Neural Network built.
print "Predicting..."
s=f=0
print "------------------------------------------------------------\n\n"
#gnb=GaussianNB()
#gnb.fit(trainSparseMatrix,column_or_1d(targetRating))
#print "GNB Prediction rate is "+str(gnb.score(testSparseMatrix,testRating)*100)

mnb=MultinomialNB(alpha=100.005)
mnb.fit(trainSparseMatrix,column_or_1d(targetRating))
print "MNB Prediction rate is "+str(mnb.score(testSparseMatrix,testRating)*100)
#bnb=BernoulliNB()
#bnb.fit(trainSparseMatrix,column_or_1d(targetRating))
#print "BNB Prediction rate is "+str(bnb.score(testSparseMatrix,testRating)*100)
#print "Cross Validation score : "+str(cvs(bnb,X=crossValidationTrain, y=crossValidationTrainResults,cv=5))
#print "Cross Validation score : "+str(cvs(gnb,X=crossValidationTrain, y=crossValidationTrainResults,cv=5))
#print "Cross Validation score : "+str(cvs(mnb,X=crossValidationTrain, y=crossValidationTrainResults,cv=5))
## Predict test data based on the Neural Network formed from training dataset
#predictedTestResults=list(knc_map.predict(testReview))
#print "Percentage of data that is predicted CORRECT is : "+str(float(s)/len(predictedRating)*100.0)
#print "Percentage of data that is predicted WRONG is : "+str(float(f)/len(predictedRating)*100.0)

#print "\n\n------------------------------------------------------------"

#while True:
#    ans=raw_input("Do you want to predict a review?(Y/N) ").strip()
#    if ans=='Y' or ans=='y':
#        reviewToPredict=[]
#        r1=raw_input("Enter the review to predict rating: \n")
#        reviewToPredict.append(str(r1).lower())
#        predictionSparseMatrix=prepareSparseMatrix(convertReviews(reviewToPredict))
#        predictionDataFeatures = pd.DataFrame(predictionSparseMatrix,columns=decisionAttributes)
#        print str(list(knc_map.predict(predictionDataFeatures)))
#    elif ans=='N' or ans=='n':
#        print "Exitting.."
#        exit(0)
#    else:
#        print "Invalid Input"
