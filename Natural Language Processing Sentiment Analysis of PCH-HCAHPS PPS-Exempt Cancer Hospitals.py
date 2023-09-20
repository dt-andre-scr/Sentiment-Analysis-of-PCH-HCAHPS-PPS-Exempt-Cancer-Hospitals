
# data available for download at --> https://data.cms.gov/provider-data/dataset/qatj-nmws <--  last edited: 09/19/2023

# larger dataset to use, but huge for personal projects --> https://data.cms.gov/provider-data/dataset/dgck-syfz <--
# ** I have not written the program for the larger data set**


#roBERTa NLP Model

#you might need to install the below :
#!pip install transformers
#!pip install torch
#-------------------------------------#
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
import numpy as np


df= pd.read_csv("insert_your_path_here/PCH_HCAHPS_STATE.csv", delimiter = ',')

text = df['HCAHPS Answer Description'] #these are the patient responses, we will isolate them here

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#creating function for NLP processing
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

#creating empty list and loop for polarity aka sentiment scores attained from model
sentiment_scores = []

#looping through the isolated series and pushing each instance through the NLP solver
for i, row in tqdm(text.iteritems(), total = len(text)):
    
    roberta_scores = polarity_scores(row)
    sentiment_scores.append(roberta_scores)
    
#checkpoint
#sentiment_scores

#putting sentiment scores into a dataframe
sentiment_scores = pd.DataFrame(sentiment_scores)

#checkpoint
sentiment_scores

#giving df a new column named Id to prepare for merge
df = df.reset_index().rename(columns = {'index': 'Id'})
df

#giving sent_scores a new column named Id to prepare for merge
sentiment_scores = sentiment_scores.reset_index().rename(columns = {'index': 'Id'})
sentiment_scores

#using the newly created Id column to merge both dataframes using a left join
df_final = df.merge(sentiment_scores, how = 'left')
df_final

#exporting the df_final dataframe as a csv
df_final.to_csv("insert_your_path/name_your_csv.csv")

#**adding more processing for deeper analysis, will not be my final --currently unfinished--** 09/19/2023