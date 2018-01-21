#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Read in the data from our training dataset
file = os.path.dirname(__file__) + '/../Archive/SentimentAnalysisDataset.csv'
df = pd.read_csv(file,  error_bad_lines=False)

# Exploring the positive Vs the negative tweets
print('Mean of training dataset sentiment', df['Sentiment'].mean())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['SentimentText'],
                                                    df['Sentiment'],
                                                    random_state=0)

print(' A sample from X_train set:\n\n', X_train.iloc[1], '\n\n X_train size: ', X_train.shape, "\n\n X_test size:", X_test.shape)


# Fit the CountVectorizer to the training data
vectorizer = CountVectorizer().fit(X_train)
print('Number  of features inside the document :', len(vectorizer.get_feature_names()))

# transform the documents in the training data to a document-term matrix
X_trainvectorized = vectorizer.transform(X_train)

# Train the model
classification_model = LogisticRegression()
classification_model.fit(X_trainvectorized, y_train)

# Predict the transformed test documents
predictions = classification_model.predict(vectorizer.transform(X_test))

print('ROC AUC: ', roc_auc_score(y_test, predictions))

# get the feature names as numpy array
features = np.array(vectorizer.get_feature_names())

# Sort the coefficients from the model
coefficient_index = classification_model.coef_[0].argsort()

# print 10 smallest and 10 largest coefficients ( negative and positive Tokens )
print('Tokens with large coefficients (Positive_tokens): {}'.format(features[coefficient_index[:-11:-1]]))
print('Tokens with small coefficients (negative_tokens):{}\n'.format(features[coefficient_index[:10]]))

'-----------------------------------------------------------------------------------------------'
'-------------------- improving the model with ngrams and tf-idf --------------------------------'

new_vectorizer = TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2),tokenizer=nltk.TweetTokenizer().tokenize).fit(X_train)
print('Number of features inside the document :', len(new_vectorizer.get_feature_names()))

X_trainvectorized = new_vectorizer.transform(X_train)

classification_model = LogisticRegression()
classification_model.fit(X_trainvectorized, y_train)

predictions = classification_model.predict(new_vectorizer.transform(X_test))

print('ROC AUC: ', roc_auc_score(y_test, predictions))

features = np.array(new_vectorizer.get_feature_names())

coefficient_index = classification_model.coef_[0].argsort()

print('Tokens with small coefficients (negative_tokens):{}\n'.format(features[coefficient_index[:10]]))
print('Tokens with large coefficients (Positive_tokens): {}'.format(features[coefficient_index[:-11:-1]]))


'-------------------------------------------------------------------------------------------------------'
'--------------------------------- testing the model against our real dataset ---------------------------'

print(classification_model.predict(new_vectorizer.transform(['1 hour to get to work today, my wife is only 1/2 way to work so about 2 hours for her.  Glad the city decided to shut off the 20 again',
                                                             "IT'S COMING... exactly what you need to make it to the next step of your journey.  Stay tuned.",
                                                             "If you are yourself you are BEAUTIFUL",
                                                             "help looks like winter has decided to stay",
                                                             "Thank you for leading an amazing workshop @kpatelneha23 !",
                                                             "Artificial Intelligence to Transform Workplace Sooner Than Expected"])))