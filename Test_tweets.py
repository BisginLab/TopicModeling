import os
import re
import pandas as pd
from bertopic import BERTopic

# Set data path
artifacts_dir = r"C:\Users\DELL\Desktop\CSC501\OneDrive-2024-06-06\Data of tweets"
input_path = os.path.join(artifacts_dir, "all_tweets_delimiter_cleaned.csv")

# Load data and basic cleaning
trump = pd.read_csv(input_path)
trump['content'] = trump['content'].apply(lambda row: re.sub(r"http\S+", "", row).lower())
trump['content'] = trump['content'].apply(lambda row: " ".join(filter(lambda x: x[0] != "@", row.split())))
trump['content'] = trump['content'].apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row).split()))

# Extract timestamps and tweet content
timestamps = trump['date'].to_list()
tweets = trump['content'].to_list()

# Create and train BERTopic model
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(tweets)

# Generate topic representations over time
topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)

# Debugging output
print("Visualization start")

# Display chart for top 20 topics
fig1 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=50)
fig1.show()

print("Visualization end")
