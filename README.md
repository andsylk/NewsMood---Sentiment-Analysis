
### News Mood Analysis

#### Summary of Project
Using Python, Twitter API, I will perform a sentiment analysis of the Twitter activity of the various news following news organizations: BBC, CBS, CNN, Fox, and New York Times and then present my findings visually.


#### Observed Trends

#### 1) The average compound score for all the news organizations show a neutral sentiment with average compound scores ranging from -0.08 to 0.40.
#### 2) CBS and CNN had a much higher average compound score than the other news organizations due to tweets and/or retweets about sports and winners of the Golden Globe awards.
#### 3) Only 10.4% of all tweets had a negative sentiment, with the majority of tweets (60%) having a neutral sentiment.


```python
# Dependencies
import pandas as pd
import tweepy
import json
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import seaborn as sns
```


```python
# Twitter API Keys
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target News Organizations
target = ["@BBCWorld", "@nytimes", "@foxnews", "@cbs", "@cnn"]
```


```python
overall_sentiment = []
sentiments = []

# Loop through all target news sources
for target_user in target:

    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    counter = 1
    
    for x in range(1):
    
        public_tweets = api.user_timeline(target_user, count=100, result_type="recent")

        #Loop through all tweets
        for tweet in public_tweets:
            
        # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            tweets_ago = counter

            # Add each value to the appropriate array
            sentiments.append({"News Org": tweet["user"]["screen_name"],
                               "Date": tweet["created_at"],
                               "Tweet": tweet["text"],
                               "Compound": compound,
                               "Positive": pos,
                               "Negative": neu,
                               "Neutral": neg,
                                "Tweets Ago": counter})
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            counter = counter + 1

    # Store the Overall Average Sentiments
    overall_sentiment.append({"News Org": target_user,
                             "Compound": np.mean(compound_list),
                             "Positive": np.mean(positive_list),
                             "Neutral": np.mean(negative_list),
                             "Negative": np.mean(neutral_list),
                             "Tweet Count": len(compound_list)})

```


```python
# Create Data Frame from dictionary
sentiments1 = pd.DataFrame.from_dict(sentiments)
sentiments_pd = sentiments1[["News Org", "Date","Tweet","Compound","Positive","Negative","Neutral","Tweets Ago"]]
```


```python
# Clean up Data Frame and export to CSV file
sentiments_pd["News Org"].replace("BBCWorld", "BBC", inplace=True)
sentiments_pd["News Org"].replace("nytimes", "New York Times", inplace=True)
sentiments_pd["News Org"].replace("FoxNews", "Fox", inplace=True)
sentiments_pd.head()
sentiments_pd.to_csv('news_org_sentiment_analysis.csv')
```


```python
# Create Data Frame for Overall Sentiments
overall_sentiment1 = pd.DataFrame.from_dict(overall_sentiment)
compound_sentiment = overall_sentiment1[["News Org","Compound"]]
compound_sentiment["News Org"].replace("@BBCWorld", "BBC", inplace=True)
compound_sentiment["News Org"].replace("@nytimes", "New York Times", inplace=True)
compound_sentiment["News Org"].replace("@foxnews", "Fox", inplace=True)
compound_sentiment["News Org"].replace("@cbs", "CBS", inplace=True)
compound_sentiment["News Org"].replace("@cnn", "CNN", inplace=True)
#compound_sentiment
```

```python
# Set style of scatterplot
sns.set_style("ticks")
plt.style.use("seaborn")


# Create scatterplot using Seaborn
sns.lmplot(x ="Tweets Ago", 
           y ="Compound",
           data=sentiments_pd, 
           hue="News Org",
           fit_reg=False,
           size = 6,
           aspect = 1.5,
           scatter_kws={"marker": "D",
                        "s": 80,
                      "edgecolor":sns.xkcd_rgb["black"],
                      "linewidth": 1})

# Set title, x and y labels, sizing, and formatting of chart
plt.title("Sentiment Analysis of News Org Tweets (01/07/2018)", fontsize = 18, fontweight='bold')
plt.xlabel("Tweets Ago", labelpad=10, fontsize = 14)
plt.ylabel("Tweet Polarity",fontsize = 14)
plt.subplots_adjust(top=0.88)
plt.xticks(size = 12)
plt.yticks(size = 12)

# Save image and display image
plt.savefig("sentiment_analysis_scatter.png")
plt.show()
```



```python
# Plot set-up of x-axis and colors and size
x_axis = np.arange(len(compound_sentiment))
fig = plt.figure(figsize = (9,7))
bar_color = ["darkblue", "darkseagreen","firebrick","purple","khaki"]

# Creat bar chart 
plt.bar(x_axis, compound_sentiment["Compound"],alpha=0.6, align="edge",width=1, 
        linewidth=2, edgecolor="white",color=bar_color )

# Specify tick location and labels
tick_locations = [value+ 0.5 for value in x_axis] #+0
plt.xticks(tick_locations, ["BBC", "New York Times", "Fox", "CBS", "CNN"])


# Set title, x and y labels, sizing, and formatting of chart
plt.title("Overall Media Sentiment based on Twitter (01/07/2018)", fontsize = 18, fontweight='bold')
plt.xlabel("News Org", labelpad=20, fontsize = 14)
plt.ylabel("Tweet Polarity", fontsize = 14)
plt.axhline(0, color="black")
plt.xticks(size = 11)
plt.yticks(size = 12)

for a,b in zip(x_axis, compound_sentiment["Compound"]):
    b = round(b,2)
    if b < 0:
        plt.text(a+0.3, b +0.05, str(b), color='black',fontsize = 13)
    else:
        plt.text(a+0.38, b-0.03, str(b), color='black',fontsize = 13)

# Save image and display image
plt.savefig("overall_sentiment.png")
plt.show()
```


