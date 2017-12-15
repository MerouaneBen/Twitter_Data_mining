#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords

# Exploring tweet content
file = os.path.dirname(__file__) + '../Archive/twitter_data_set.json'

with open(file, 'r', encoding='utf-8') as tweets_file:
    data = tweets_file.read()
    tweets = json.loads(data)
    print(json.dumps(tweets[1], indent=4, ensure_ascii=False))


# word_tokenize example
tweet = '@Merouane_Benth: This is just a tweet example! #NLTK :) http://www.twitter.com'
print(word_tokenize(tweet))


# TweetTokenizer example
tokenizer = TweetTokenizer()
tokens = tokenizer.tokenize(tweet)
print(tokens)

# removing punctuation
emoticons_str = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""
regex_str = [
    emoticons_str,
    r'(?:@[\w_]+)', # @mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hashtags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'] # URLs

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

tweet_tokens_list = ['@Merouane_Benth', ':', 'This', 'is', 'just', 'a', 'tweet', 'example', '!', '#NLTK', ':)', 'http://www.twitter.com']
clean_token_list = []

for element in tweet_tokens_list:
    if tokens_re.findall(element):
        clean_token_list.append(element)
    else:
        if not re.match(r'[^\w\s]', element):
            clean_token_list.append(element)

print("Original token list:", tweet_tokens_list)
print("New token list:", clean_token_list)

# Loading stop words
stop_words_english = stopwords.words('english')
stop_words_french = stopwords.words('french')
print(' English stop words:', stop_words_english, '\n', 'French stop words:', stop_words_french)

# Filtering out stop words
words = [w.lower() for w in clean_token_list if not w.lower() in stop_words_english]
print(words)
