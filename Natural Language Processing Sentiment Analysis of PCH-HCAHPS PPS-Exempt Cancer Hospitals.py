# data available for download at --> https://data.cms.gov/provider-data/dataset/qatj-nmws <--  last edited: 09/25/2023
# larger dataset to use if you'd like to --> https://data.cms.gov/provider-data/dataset/dgck-syfz <--
# ** I have not written the program for the larger data set**

#roBERTa & vader NLP Model Ensemble

#you might need to install the below :
#!pip install transformers
#!pip install torch
#-------------------------------------#
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#file we will be using to extract the data from
df= pd.read_csv("insert_your_path_here/PCH_HCAHPS_STATE.csv", delimiter = ',')

#vader model setup and download
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

#these are the patient responses
text = df['HCAHPS Answer Description'] 

#roBERTa model setup
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#creating function for roBERTa NLP processing
def polarity_scores(example):
    
    encoded = tokenizer(example, return_tensors='pt')
    output = model(**encoded)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {

        'negative' : scores[0],
        'neutral' : scores[1],
        'positive' : scores[2]

    }
    return scores_dict

#creating empty lists to append sentiment scores from functions
sentiment_scores = []
vader_scores = []

#looping through the text dataframe to apply NLP processing
#vader NLP processing comes from sia.polarity_scores()
for i, row in tqdm(text.iteritems(), total = len(text)):
    
    vader_score = sia.polarity_scores(row)
    vader_scores.append(vader_score)
    roberta_scores = polarity_scores(row)
    sentiment_scores.append(roberta_scores)
    
#checkpoint
#sentiment_scores
#vader_scores

#formatting lists into dataframes
sentiment_scores = pd.DataFrame(sentiment_scores)
vader_scores = pd.DataFrame(vader_scores)

#giving df a new column named Id to prepare for merge
df = df.reset_index().rename(columns = {'index': 'Id'})
df

#giving sentiment_scores a new column named Id to prepare for merge
sentiment_scores = sentiment_scores.reset_index().rename(columns = {'index': 'Id'})
sentiment_scores

#giving vader_scores a new column named Id to prepare for merge
vader_scores = vader_scores.reset_index().rename(columns = {'index': 'Id'})
vader_scores

#first join on Id with main df and sentiment_scores
df_join1 = df.merge(sentiment_scores, how = 'left')

#second join on Id with df_join1 and vader_scores
df_join2 = df_join1.merge(vader_scores, how = 'left')
df_join2

#creating new dataframe from df_final to use for analysis
df_final = df_join2[['State', 'HCAHPS Answer Description', 'negative', 'neutral', \
                     'positive', 'neg', 'neu', 'pos', 'compound']]

#checkpoint
#df_final

#combining roBERTa and vader sentiments and /2 to get average of both
df_final['new_negative'] = (df_final['negative'] + df_final['neg'])/2
df_final['new_neutral'] = (df_final['neutral'] + df_final['neu'])/2
df_final['new_positive'] = (df_final['positive'] + df_final['pos'])/2

#checkpoint
#df_final

#the (?i) flag makes the search case-insensitive
#the \b ensures that the phrase is matched as a separate word
#creating the parameters for our text grouping
neg_search_string = r'(?i)\b(never|would not|did not|disagree)\b'
neu_search_string = r'(?i)\b(usually)\b'
pos_search_string = r'(?i)\b(always|agree)\b'

#applying parameters (found after analysis)..
#.. creating new columns to group text into sentiments,..
#.. using string variable and regex --we get warnings here
df_final['neg_sentiment'] = df_final['HCAHPS Answer Description'].str.contains(neg_search_string, regex=True)
df_final['neu_sentiment'] = df_final['HCAHPS Answer Description'].str.contains(neu_search_string, regex=True)
df_final['pos_sentiment'] = df_final['HCAHPS Answer Description'].str.contains(pos_search_string, regex=True)

#checkpoint
#df_final

#grouping negative responses into their own dataframe
negative_responses = df_final[df_final['neg_sentiment'] == True]

#grouping neutral responses into their own dataframe
neutral_responses = df_final[df_final['neu_sentiment'] == True]

#grouping positive responses into their own dataframe
positive_responses = df_final[df_final['pos_sentiment'] == True]

#filtering out data not caught by the text parameters into their own dataframe
#we will create an algorithm to group these
filtered_data = df_final[(df_final['neg_sentiment'] == False) & (df_final['neu_sentiment'] == False)\
                         & (df_final['pos_sentiment'] == False)]

#checkpoint
#filtered_data

#parameters for assigning negative sentiment to the filtered_data
#using these parameters to put them into their own dataframe
filtered_data_neg = filtered_data[(filtered_data['new_negative'] > 0.1) & \
                                          (filtered_data['negative'] > filtered_data['positive'])]

#checkpoint
#filtered_data_neg

#parameters for assigning neutral sentiment to the filtered_data
#using these parameters to put them into their own dataframe
filtered_data_neu = filtered_data[(filtered_data['new_negative'] < 0.1) & \
                                  (filtered_data['new_positive'] < 0.1) & \
                                  (filtered_data['neu'] > 0.8)]

#checkpoint
#filtered_data_neu

#parameters for assigning positive sentiment to the filtered_data
#using these parameters to put them into their own dataframe
filtered_data_pos = filtered_data[filtered_data['new_positive'] > 0.1]

#checkpoint
#filtered_data_pos

#concat negative_responses and filtered_data_neg to gather all 
#negative sentiment responses
final_neg = pd.concat([negative_responses, filtered_data_neg])

#checkpoint
#final_neg

#concat neutral_responses and filtered_data_neu to gather all
#neutral sentiment responses
final_neu = pd.concat([neutral_responses, filtered_data_neu])

#checkpoint
#final_neg

#concat positive_responses and filtered_data_pos to gather all
#positive sentiment responses
final_pos = pd.concat([positive_responses, filtered_data_pos])

#checkpoint
#final_pos

#getting the total of negative responses as an integer
total_neg = len(filtered_data_neg)+len(negative_responses)

#checkpoint
#total_neg

#getting the total of neutral responses as an integer
total_neu = len(filtered_data_neu)+len(neutral_responses)

#checkpoint
#total_neu

#getting the total of positive responses as an integer
total_pos = len(filtered_data_pos)+len(positive_responses)

#checkpoint
#total_pos

#getting total of all responses to use for denominator
total_all = len(df_final)

#checkpoint
#total_all

#finding the percent of that are negative
neg_percent = (total_neg/total_all)*100

#checkpoint
#neg_percent

#finding the percent of that are neutral
neu_percent = (total_neu/total_all)*100

#checkpoint
#neu_percent

#finding the percent of that are positive
pos_percent = (total_pos/total_all)*100

#checkpoint
#pos_percent

#creating the visuals using plotly

#percentages for the negative bar graph
neg_graph = go.Bar(x=['NEGATIVE'], y=[neg_percent], name='NEGATIVE')

#percentages for the neutral bar graph
neu_graph = go.Bar(x=['NEUTRAL'], y=[neu_percent], name='NEUTRAL')

#percentages for the positive bar graph
pos_graph = go.Bar(x=['POSITIVE'], y=[pos_percent], name='POSITIVE')

#combine the traces in a list
data = [neg_graph, neu_graph, pos_graph]

#set the layout
layout = go.Layout(title='Percentage Bar Graphs')

#create the figure and plot
fig = go.Figure(data=data, layout=layout)
fig.update_layout(barmode='group', xaxis=dict(type='category'))

#show graph here if you'd like
fig.show()

#dropping columns to clean up the dataframe
final_neg.drop(['neg_sentiment', 'neu_sentiment', 'pos_sentiment'], axis=1, inplace=True)

#checkpoint
#final_neg

#dropping columns to clean up the dataframe
final_neu.drop(['neg_sentiment', 'neu_sentiment', 'pos_sentiment'], axis=1, inplace=True)

#checkpoint
#final_neu

#dropping columns to clean up the dataframe
final_pos.drop(['neg_sentiment', 'neu_sentiment', 'pos_sentiment'], axis=1, inplace=True)

#checkpoint
#final_pos

#this is to export the first pass of the analysis for if you want to check on it later
#df_final.to_csv("insert_your_path_here/data_health_testing.csv")

#we are exporting the results of each sentiment into different files:

#results of all negative responses
final_neg.to_csv("insert_your_path_here/final_neg.csv")

#results of all nuetral responses
final_neu.to_csv("insert_your_path_here/final_neu.csv")

#results of all positive responses
final_pos.to_csv("insert_your_path_here/final_pos.csv")
