import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def loadData(isTrain):
    """
    Load test and train data
    """
    csv = "test.csv"
    if(isTrain):
        csv = "train.csv"
    return pd.read_csv(csv)


def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    #nopunc = [char for char in text if char not in string.punctuation]

    stemmer = PorterStemmer()

    nopunc = []
    stopwordsRemoved = []

    for char in text:
        if(char not in string.punctuation):
            nopunc.append(char)

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    for word in nopunc.split():
        word = word.lower()
        if word not in stopwords.words('english'):
            word = stemmer.stem(word);
            stopwordsRemoved.append(word)

    return stopwordsRemoved


def buildPipeline(model):
    return Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', model), # train on TF-IDF vectors w/ Naive Bayes classifier
    ])


def main():
    trainData = loadData(True)
    testData = loadData(False)

    txt_train = trainData['text']
    label_train = trainData['author']

    txt_test = testData['text']
    
    #model = SVC(kernel = 'rbf', probability = True)
    
    
    model = RandomForestClassifier(max_features='sqrt', oob_score = True) 
 
    # Use a grid over parameters of interest
    para = { 
           "classifier__n_estimators" : [10,50, 100, 500],
           "classifier__max_depth" : [1, 10, 25, 30],
           "classifier__min_samples_leaf" : [1, 2, 6, 10]
           }
    
    
    
    pipeline = buildPipeline(model)

    k_fold = KFold(n_splits=5)

 
    grid_search = GridSearchCV(estimator = pipeline, param_grid = para, cv = k_fold, n_jobs = -1)
    

    
    grid_search.fit(txt_train,label_train)

    predictions = grid_search.predict_proba(txt_test)
    
    print("***************************************")
    print(grid_search.best_params_)
    print("The best classifier is: ", grid_search.best_estimator_)
    print("***************************************")

    sub = pd.DataFrame(predictions, columns=['EAP', 'HPL', 'MWS'])

    iddd = pd.DataFrame(testData['id'], columns=['id'])

    sub = iddd.join(sub)

    sub.to_csv("submission.csv")



if __name__ == "__main__": main()





#******** Output ******
#***************************************
#{'classifier__n_estimators': 500, 'classifier__max_depth': 30, 'classifier__min_samples_leaf': 1}
#The best classifier is:  Pipeline(memory=None,
#     steps=[('bow', CountVectorizer(analyzer=<function text_process at 0x1a18cf2d90>,
#        binary=False, decode_error='strict', dtype=<class 'numpy.int64'>,
#        encoding='utf-8', input='content', lowercase=True, max_df=1.0,
#        max_features=None, min_df=1, ngram_range=(1, 1), preprocessor=None...imators=500, n_jobs=1,
#            oob_score=True, random_state=None, verbose=0, warm_start=False))])
#***************************************