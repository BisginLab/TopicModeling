# Ryan Madar
# Sometime in March 2024
# Designed to clean a particular dataset containing tweets surrounding the 2023 Turkey–Syria earthquakes


import pandas as pd
import re
import os

# Directory path for full code. Change to where your dataset is located.
directory_path = r"C:\Users\rmada\OneDrive\Documents\School Work\2024\Winter\research\Data Cleanup Tweets"
# Define the file name. If dataset changes, change the name here
dataset_name = "all_tweets_delimiter1.csv"


# Name of the file to store the cleaned characters
cleaned_characters_name="cleaned_characters.txt"
# Join the directory path with the file name to get the full file path
file_path = os.path.join(directory_path, dataset_name)
# Create a set to store the unique characters that are removed
cleaned_characters = set()

# Check if the original dataset exists
if not os.path.exists(file_path):
    print("Error: The specified file does not exist.")
else:
    tweets_df = pd.read_csv(file_path)

    # This function removes a  range of emojis and symbols by matching Unicode ranges
    def remove_emojis_and_symbols(text):
        global cleaned_characters
        emoji_pattern = re.compile("["
                                "\U0001F100-\U0001F1FF"  # Enclosed Alphanumeric Supplement
                                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                                "\U0001F600-\U0001F64F"  # emoticons
                                "\U0001F680-\U0001F6FF"  # transport & map symbols
                                "\U0001F700-\U0001F77F"  # Alchemical symbols
                                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                "\U00002702-\U000027B0"  # Dingbats
                                "\U0001F650-\U0001F67F"  # Ornamental Dingbats
                                "\U00002500-\U00002BEF"  # various symbols (including emoticons)
                                "\U00002B05-\U00002B07"  # arrows
                                "\U00002B1B-\U00002B1C"  # squares
                                "\U00002B50"             # star
                                "\U00002B55"             # circle
                                "\U00002300-\U000023FF"  # Miscellaneous Technical
                                "\U0000FE0F"             # Variation Selectors
                                "\U00002022"             # Bullet
                                "]+", flags=re.UNICODE)
        matches = emoji_pattern.findall(text)
        cleaned_characters.update(matches)
        return emoji_pattern.sub(r'', text)


    # Extract URLs
    def extract_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        found_urls = url_pattern.findall(text)
        modified_text = url_pattern.sub('', text)
        extracted_url = found_urls[0] if found_urls else ''
        return modified_text, extracted_url

    # Identified certain numeric emoji representations that required additional regex pattern matching to remove
    def remove_keycap_emojis(text):
        keycap_pattern = re.compile(r'\d\u20E3')
        return keycap_pattern.sub(r'', text)
    
    # Extract hashtags
    def extract_hashtags(text):
        hashtag_pattern = re.compile(r'\#\w+')
        found_hashtags = hashtag_pattern.findall(text)
        hashtags_string = ', '.join(found_hashtags)
        return hashtags_string
    tweets_df['Hashtags'] = tweets_df['content'].apply(extract_hashtags)

    # Extract Mentions
    def extract_mentions(text):
        mention_pattern = re.compile(r'@\w+')
        found_mentions = mention_pattern.findall(text)
        mentions_string = ', '.join(found_mentions)
        return mentions_string
    tweets_df['mentions'] = tweets_df['content'].apply(extract_mentions)


    # TIME TO CLEAN

    # Remove emojis and symbols from the content column
    tweets_df['content'] = tweets_df['content'].apply(remove_emojis_and_symbols)
    # Remove keycap emojis from the content column
    tweets_df['content'] = tweets_df['content'].apply(remove_keycap_emojis)
    # Remove new line and carriage returns
    tweets_df['content'] = tweets_df['content'].str.replace(r'\n|\r', ' ', regex=True)
    # Extract URLs from the content column and stores in a new column
    tweets_df['content_urls'] = tweets_df['content'].apply(lambda x: extract_urls(x)[1])
    # Removes URLs from the content column
    tweets_df['content'] = tweets_df['content'].apply(lambda x: extract_urls(x)[0])
    # Trim leading and trailing whitespace from the 'content' column
    tweets_df['content'] = tweets_df['content'].str.strip()
    # Remove rows where 'content' is now either empty or null
    tweets_df = tweets_df.dropna(subset=['content'])
    tweets_df = tweets_df[tweets_df['content'] != '']


    # Designed to increment a counter until a unique filename is found
    # Since this involved a lot of re-running to ensure I cleaned all the data, I made it so that the cleaned data is saved to a new file and does not overwrite the previous file
    cleaned_file_base = "all_tweets_delimiter_cleaned"
    cleaned_file_ext = ".csv"
    cleaned_file_path = os.path.join(directory_path, cleaned_file_base + cleaned_file_ext)
    counter = 1
    # If the file already exists, increment the counter until a unique filename is found
    while os.path.exists(cleaned_file_path):
        cleaned_file_path = os.path.join(directory_path, f"{cleaned_file_base}_{counter}{cleaned_file_ext}")
        counter += 1

    # Save the cleaned dataframe to the new CSV file
    tweets_df.to_csv(cleaned_file_path, index=False)

    # Print path to the cleaned file
    print(f"Data cleaned and saved to: {cleaned_file_path}")



    # Save the cleaned characters to a text file for dataset verification. Want to ensure we did not remove characters that should not have been removed
    cleaned_characters_path = os.path.join(directory_path, cleaned_characters_name)
    with open(cleaned_characters_path, 'w', encoding='utf-8') as f:
        for char in sorted(cleaned_characters):
            f.write(f"{char}\n")


