#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

count_tokens = Counter()


def print_top_tokens_frequencies(list_tokens, name, n=10):
    """
    this function prints the top (n) tokens
    :param list_tokens: list of tokens
    :param name: name of list to print
    :param n: number to tokens to print, default 10
    :return: None
    """

    tokens_count = Counter(list_tokens)

    # The top n tokens
    print("The Top {n} ".format(n=n) + name)
    for token, count in tokens_count.most_common(n):
        print("{0}: {1}".format(token, count))


def print_top_user_followers(list_tweets, name, n=10):
    """
    this function prints the user follower
    :param list_tweets: list 0f tweet
    :param name: name of list
    :param n: top followers, default 10
    :return: None
    """

    dict_users = {}

    for tweet in list_tweets:
        dict_users['@'+tweet['user']['screen_name']] = tweet['user']['followers_count']

    followers_count = Counter(dict_users)
    # The top n tokens
    print("The Top {n} ".format(n=n) + " users with bigest bumber of " + name)
    for token, count in followers_count.most_common(n):
        print("{0}: {1}".format(token, count))


def print_top_retweeted_posts(list_tweets, name, n=10):
    """
    this function prints the top (n) retweeted posts
    :param list_tweets: list 0f tweet
    :param name: name of list
    :param n: top posts, default 10
    :return: None
    """

    dict_retweeted = {}

    for tweet in list_tweets:
        dict_retweeted['@' + tweet['user']['screen_name'] + '  tweet_id: ' + tweet['id_str']] = tweet['retweet_count']

    retweet_count = Counter(dict_retweeted)
    # The top n tokens
    print("The Top {n} ".format(n=n) + " posts " + name)
    for token, count in retweet_count.most_common(n):
        print("{0}: {1}".format(token, count))


# Exploring tweet content
file = os.path.dirname(__file__) + '/../Archive/twitter_data_set.json'

with open(file, 'r', encoding='utf-8') as tweets_file:
    data = tweets_file.read()
    tweets = json.loads(data)

    # TweetTokenizer
    tokenizer = TweetTokenizer()

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
        r'(?:@[\w_]+)',  # @mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hashtags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+']  # URLs

    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

    for tweet in tweets:
        tweet['tokens'] = tokenizer.tokenize(tweet['text'])

        for token in tweet['tokens']:
            if tokens_re.findall(token):
                pass
            else:
                if not re.match(r'[^\w\s]+', token):
                    pass
                else:
                    tweet['tokens'].remove(token)

    list_tweets_en = [tweet for tweet in tweets if tweet['lang'] == 'en']
    list_tweets_fr = [tweet for tweet in tweets if tweet['lang'] == 'fr']

    # Loading stop words
    stop_words_english = stopwords.words('english')
    stop_words_french = stopwords.words('french')

    # Filtering out stop words
    list_tweets_with_profiles = []
    list_tokenized_tweets = []
    list_tokens = []
    list_clean_tokens= []
    list_hashtags = []
    list_mentions = []
    for tweet in list_tweets_en:
        tweet['final_tokens'] = [w.lower() for w in tweet['tokens'] if not w.lower() in stop_words_english]
        print(tweet['text'])
        list_tokenized_tweets.append([tok_en for tok_en in tweet['final_tokens'] if
                                      not tok_en[0] == '@' and not tok_en[:4] == 'http' and not tok_en[
                                                                                                    0] == '#' and not tok_en == 'rt' and tok_en.isalpha() == True])
        list_tweets_with_profiles.append(
            ['@' + tweet['user']['screen_name'], tweet['id_str'], tweet['text'].replace('\n', ' ').replace('\t', ' ')])
        tweet['clean_tokens'] = [tok_en for tok_en in tweet['final_tokens'] if
                                 not tok_en == 'rt' and tok_en.isalpha() == True]
        for element in tweet['clean_tokens']:
            list_clean_tokens.append(element)

        for elem in tweet['final_tokens']:
            list_tokens.append(elem)
            if elem[0] == '#':
                list_hashtags.append(elem)
            elif elem[0] == '@':
                list_mentions.append(elem)

    counter = Counter(list_tokens)

    print('Count all Tokens:',counter, '\n')
    print_top_tokens_frequencies(list_tokens, name='Tokens')
    print('\n')
    print_top_tokens_frequencies(list_clean_tokens, name='Clean Tokens')
    print('\n')
    print_top_tokens_frequencies(list_hashtags, name='Hashtags')
    print('\n')
    print_top_tokens_frequencies(list_mentions, name='Mentions')
    print('\n')
    print_top_user_followers(list_tweets_en, 'followers')
    print('\n')
    print_top_retweeted_posts(list_tweets_en, 'retweeted')

    '----------------------------------------------------------------------'
    '---------------- Topic modelling LDA ---------------------------------'

    # Create p_stemmer of class PorterStemmer
    stemmer = PorterStemmer()

    texts = []
    # stem token
    for tokenized_tweet in list_tokenized_tweets:
        temed_tokens = [stemmer.stem(i) for i in tokenized_tweet]
        texts.append(temed_tokens)

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=7, id2word=dictionary, passes=20)

    print(ldamodel.print_topics(num_topics=7, num_words=4))
