import json


geo_data = {
    "type": "FeatureCollection",
    "features": []
}
with open('tweets2.json') as f:
    for line in f:
        tweets = json.loads(line)
        for tweet in tweets:
            if tweet['coordinates']:
                geo_json_feature = {
                    "type": "Feature",
                    "geometry": tweet['coordinates'],
                    "properties": {
                        "text": tweet['text'],
                        "created_at": tweet['created_at']
                    }
                }
                geo_data['features'].append(geo_json_feature)

# Save geo data
with open('geo_data_3.json', 'w') as fout:
    fout.write(json.dumps(geo_data, indent=4))
