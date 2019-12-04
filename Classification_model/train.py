import pandas as pd
import re
import numpy as np 
import string
import nltk
import warnings 
from nltk.stem.porter import * 
# nltk.download('stopwords')
# nltk.download('punkt')
from sklearn.svm import SVC
import pickle 
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

stopword_list = nltk.corpus.stopwords.words('english')

#Text Pre_processing
def pre_processing(text):
    
    expr = re.compile('\d{2}/\d{2}/\d{4}')
    line = re.sub(expr,'',text )
    expr = re.compile('\d{2}:\d{2}:\d{2}')
    text = re.sub(expr, '', line)
    text = re.sub("\d", "", text)
    tokens = nltk.word_tokenize(text)
    pattern = re.compile('[!"\\#\\$%\\&\'\\(\\)\\*\\+,\\-\\.:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]')
    tokens = [w for w in tokens if w not in stopword_list]
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
 
    return filtered_text


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

def integer_convert(predictions):
    
    int_predict = []
    
    for i in range(0,len(predictions)):
        
        result = np.where(predictions[i] == np.amax(predictions[i]))

        int_predict.append(result[0][0])
        
    return int_predict

def training(train):
    print('Training_started\n')
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train.labels.values)
    filename = 'weights/lbl_enc.sav'
    pickle.dump(lbl_enc, open(filename, 'wb'))
    
    lbl_enc.inverse_transform(y)

    x = train['CASE_HISTORY']

    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, 
                                                      stratify=y, 
                                                      random_state=42, 
                                                      test_size=0.1, shuffle=True)

    
    
    # Always start with these features. They work (almost) everytime!
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv =  tfv.transform(xtrain) 
    xvalid_tfv = tfv.transform(xvalid)
    filename = 'weights/tfv_weights.sav'
    pickle.dump(tfv, open(filename, 'wb'))
    
    # Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain_tfv)
    xtrain_svd = svd.transform(xtrain_tfv)
    xvalid_svd = svd.transform(xvalid_tfv)
    filename = 'weights/svd_weights.sav'
    pickle.dump(svd, open(filename, 'wb'))
    
    
    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)
    filename = 'weights/scl_weights.sav'
    pickle.dump(scl, open(filename, 'wb'))
    
    
    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True) # since we need probabilities
    clf.fit(xtrain_svd_scl, ytrain)
#     joblib.dump(clf, 'level_1.pkl')
    predictions = clf.predict_proba(xvalid_svd_scl)
    
#     print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    int_predict = integer_convert(predictions)
    print('Accuracy: %0.3f' % accuracy_score(yvalid, int_predict))
    
    filename = 'weights/clf_weights.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Modle weights saved in weights folder")
    
if __name__ == "__main__": 
    
    df1 = pd.read_csv("Input_data/input.csv")
    
     
    print('\nPre_processing_started!!!!\n')
    clean = []
    for i in range(0,len(df1)):
        
        clean.append(pre_processing(df1['CASE_HISTORY'][i]))
                     
    df = pd.DataFrame(columns = ['clean','CASE_HISTORY','labels'])
    
    df['clean'] = clean
     
    df['CASE_HISTORY'] = df1['CASE_HISTORY']
    
    df['labels'] = df1['CLASS']
        
    training(df)
    
    
    
    
