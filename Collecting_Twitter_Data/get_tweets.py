#!/usr/bin/env python
# -*- coding: utf-8 -*-

from twython import Twython
import configparser
import sys
import json
import os

config = configparser.ConfigParser()

# reading  APP_KEY and APP_SECRET from the config.ini file
# update with your credentials before running the app
config.read(os.path.dirname(__file__) + '../config.ini')

APP_KEY = config['twitter_connections_details']['APP_KEY']
APP_SECRET = config['twitter_connections_details']['APP_SECRET']

api = Twython(APP_KEY, APP_SECRET)

if not api:
    print('Authentication failed!.')
    sys.exit(-1)

tweets = []   # list where to store data
tweets_per_request = 100
max_tweets_to_be_fetched = 10000   # number of tweets you want fetch (optional)

search_query = "geocode:45.533471,-73.552805,12mi since:2017-12-01"

max_id = None
since_id = None
count_tweets = 0

while count_tweets < max_tweets_to_be_fetched:    # we continue our calls till we get the number of tweets desired
    try:
        if max_id:   # for the first page of result received (max_id = 0)
            if not since_id:
                tweets_fetched = api.search(q=search_query, count=tweets_per_request)
            else:
                tweets_fetched = api.search(q=search_query, count=tweets_per_request, since_id=since_id)
        else:           # for the following pages max_id = some number
            if not since_id:
                tweets_fetched = api.search(q=search_query, count=tweets_per_request, max_id=max_id)
            else:
                tweets_fetched = api.search(q=search_query, count=tweets_per_request, max_id=max_id, since_id=since_id)

        if not tweets_fetched:
            print("No more tweets found")
            break

        for tweet in tweets_fetched["statuses"]:
            tweets.append(tweet)

        count_tweets += len(tweets_fetched['statuses'])

        sys.stdout.write('\r Number of downloaded Tweets: {0} '.format(count_tweets))

        # get the max_id
        max_id = tweets_fetched['search_metadata']["max_id_str"]

    except Exception as e:
        # Just exit if any error
        print("some error : " + str(e))
        break

# writing tweets to json file
with open('twitter_data_set.json', 'w') as file:
    json.dump(tweets, file)
