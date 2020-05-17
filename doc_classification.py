import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import shutil, os

# Step 1 - Get the file details
directory = []
file = []
title = []
text = []
label = []
datapath = './Datasets/bbc/'
for dirname, _ , filenames in os.walk(datapath):
    try:
        filenames.remove('README.TXT')
    except:
        pass
    for filename in filenames:
        directory.append(dirname)
        file.append(filename)
        label.append(dirname.split('/')[-1])
        #print(filename)
        fullpathfile = os.path.join(dirname,filename)
        with open(fullpathfile, 'r', encoding="utf8", errors='ignore') as infile:
            intext = ''
            firstline = True
            for line in infile:
                if firstline:
                    title.append(line.replace('\n',''))
                    firstline = False
                else:
                    intext = intext + ' ' + line.replace('\n','')
            text.append(intext)

#creating df frames for text analysis
fulldf = pd.DataFrame(list(zip(directory, file, title, text, label)),
               columns =['directory', 'file', 'title', 'text', 'label'])

df = fulldf.filter(['title','text','label'], axis=1)

# print("FullDf : ", fulldf.shape)
# print("DF : ", df.shape)

df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.text).toarray()
labels = df.category_id

#chi2 features selection
N = 3
for label, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#   print("# '{}':".format(label))
#   print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
#   print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5, random_state=42)
]

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)

while True:
    choice = input('Choose the ML method: \n(1) Logistic Regression (2)Random Forest (3)SVM (4)Naive Bayes\n')
    if choice == '1':
        model = LogisticRegression(random_state=0)
        break
    elif choice == '2':
        model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
        break
    elif choice == '3':
        model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
        break
    elif choice == '4':
        model = MultinomialNB()
        break
    else:
        print("Please Enter a number between 1 and 4!!")


model.fit(features, labels)

#Prediction
folderPath = './Datasets/RandomArticles'

for root, _ , filelist in os.walk(folderPath):
#parse the textfile and extract the text
    for filename in filelist:
        fullpathfile = os.path.join(root,filename)
        # print(fullpathfile)
        predict_text = []
        with open(fullpathfile, 'r', encoding="utf8", errors='ignore') as infile:
            intext = ''
            firstline = True
            for line in infile:
                if firstline:
                    title.append(line.replace('\n',''))
                    firstline = False
                else:
                    intext = intext + ' ' + line.replace('\n','')
            predict_text.append(intext)
    #prediction for the given text
        text_features = tfidf.transform(predict_text)
        prediction = model.predict(text_features)
        category = id_to_category[prediction[0]]
        dest = './Datasets/bbc/'+category
        shutil.move(fullpathfile,dest)
