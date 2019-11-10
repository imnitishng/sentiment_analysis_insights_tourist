# %% [code]
import pandas as pd
import random
import os
import numpy as np
import torch

def seed_everything(seed=69):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# %% [code]
from fastai import *
from fastai.text import *

# %% [code]
bs = 32
data_clas = (TextList.from_csv('../input/imdb-fastai', 'language.csv', cols='review')
                   .split_by_rand_pct(0.1)
                   .label_from_df(cols='rating')
                   .databunch(bs=bs))

# %% [code]
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

# %% [code]
# learn.load('third')
learn.load('/kaggle/input/imdb-fastai/third');

# %% [code]
import nltk
from nltk import FreqDist
import pandas as pd

nltk.download('stopwords') # run this one time

# %% [code]
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# %% [code]
# df = pd.read_csv("../input/final-language-data/final (1).csv")
bada_new = pd.read_csv('../input/hcldatacsv2/bada_new.csv', index_col=False)

# %% [code]
bada_new.rename(columns={'caption':'review',
                      'rating':'time', 
                      'username':'rating'}, inplace=True)
del bada_new['id_review'], bada_new['timestamp'], bada_new['time'],bada_new['n_photo_user'], bada_new['url_user'], bada_new['n_review_user']

df = bada_new.drop('rating', axis=1)
# df.head()

# %% [code]
# My functions
def get_predictions(df=df):
    learn.data.add_test(df.review.values)
    preds,y = learn.get_preds(ds_type=DatasetType.Test)
    a,df['prediction'] = torch.max(preds, 1)
    df['prediction'] = df['prediction'].apply(lambda x: 'pos' if x==1 else 'neg')
    return df

def pos_neg_plot(df):
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x='prediction')
    plt.show()
    
def get_reviews_from_tag(tag):
    ret = df[df['review'].str.contains(tag)]
    pos_perc = ret[ret['prediction']=='pos'].shape[0] / ret.shape[0] * 100
    neg_perc = 100 - pos_perc
    print(f'Positive Percentage: {pos_perc}, Negative Percentage: {neg_perc}')
    return ret

def plot_negatives(s=0,e=50):
    all_words = ' '.join([text for text in df['review']])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = len(df['review'])) 
    d.reset_index(inplace=True)

    d['neg_perc'] = np.nan
    for tag in d['word'].values:
        ret = df[df['review'].str.contains(tag)]
        pos_perc = ret[ret['prediction']=='pos'].shape[0] / ret.shape[0] * 100
        neg_perc = 100 - pos_perc
        d.loc[(d['word']==tag), 'neg_perc'] = neg_perc
    d = d.sort_values('neg_perc', ascending=False)
    plt.figure(figsize=(20, 5))
    sns.barplot(data=d[s:e], x='word', y='neg_perc')
    if(e-s>60):
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=45)
    plt.xticks()
    plt.title('Percentage of Negative Reviews per tag.')
    plt.show()
    
def plot_postives(s=0,e=50):
    all_words = ' '.join([text for text in df['review']])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = len(df['review'])) 
    d.reset_index(inplace=True)

    d['pos_perc'] = np.nan
    for tag in d['word'].values:
        ret = df[df['review'].str.contains(tag)]
        pos_perc = ret[ret['prediction']=='pos'].shape[0] / ret.shape[0] * 100
        neg_perc = 100 - pos_perc
        d.loc[(d['word']==tag), 'pos_perc'] = pos_perc
    d = d.sort_values('pos_perc', ascending=False)
    plt.figure(figsize=(20, 5))
    sns.barplot(data=d[s:e], x='word', y='pos_perc')
    if(e-s>60):
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=45)
    plt.xticks()
    plt.title('Percentage of Positive Reviews per tag.')
    plt.show()
    
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in df['review']])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.xticks(rotation=45)
    plt.show()

# %% [code]
# learn.data.add_test(df.review.values)
# preds,y = learn.get_preds(ds_type=DatasetType.Test)
# a,df['prediction'] = torch.max(preds, 1)
# df['prediction'] = df['prediction'].apply(lambda x: 'pos' if x==1 else 'neg')

# %% [code]
# freq_words(df['review'], 50)

# %% [code]
# remove unwanted characters, numbers and symbols
def clean_data():    
    df['review'] = df['review'].str.replace("[^a-zA-Z#]", " ")

    # %% [code]
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    # %% [code]
    # function to remove stopwords
    def remove_stopwords(rev):
        rev_new = " ".join([i for i in rev if i not in stop_words])
        return rev_new

    # remove short words (length < 3)
    df['review'] = df['review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    # remove stopwords from the text
    reviews = [remove_stopwords(r.split()) for r in df['review']]

    # make entire text lowercase
    reviews = [r.lower() for r in reviews]

    # %% [code]
    # freq_words(reviews, 50)

    # %% [code]
    nlp = spacy.load('en', disable=['parser', 'ner'])

    def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
        output = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            output.append([token.lemma_ for token in doc if token.pos_ in tags])
        return output

    # %% [code]
    tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
    # print(tokenized_reviews[1])

    # %% [code]
    reviews_2 = lemmatization(tokenized_reviews)
    # print(reviews_2[1]) # print lemmatized review

    # %% [code]
    reviews_3 = []
    for i in range(len(reviews_2)):
        reviews_3.append(' '.join(reviews_2[i]))

    df['review_summ'] = reviews_3

# freq_words(df['review_summ'], 50)

# %% [code]
# get_reviews_from_tag('entry')
# summarize('tag')

# %% [code]
# plot_negatives()

# %% [code]
def build_lda():    
    dictionary = corpora.Dictionary(reviews_2)

    # %% [code]
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]

    # %% [code]
    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, random_state=100,
                    chunksize=1000, passes=50)

# %% [code]
    lda_model.print_topics()

# %% [code]
# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
# vis

# %% [code]
