'''
This programme is to analyze topics of discussion
within each emotion category using topic modeling.
Each .csv file has been provided based on the sentiment
analysis of discussions on Twitter and Reddit.

Columns are:
Row Number, Text, Score, Date, Link
'''

# Chapter 0: Take in User Arguments
# File will take in 1 integer argument: 1, 2 or 3
# 1 signifies that the code will present the data based on negative.csv.
# 2 signifies that the code will present the data based on neutral.csv.
# 3 signifies that the code will present the data based on positive.csv.
import sys

try:
    num = int(sys.argv[1])
except (ValueError, TypeError, NameError, IndexError):
    print('Invalid input! Please enter a valid argument')
    print('main.py <1, 2 or 3> | 1 = negative.csv, 2 = neutral.csv, 3. positive.csv')
    sys.exit()

''' 
csv = CSV file to read
wordlimit = word limit for word count graph
visoutput = output HTML for topic visualization using pyLDAvis
'''
if num == 1:
    csv = 'negative.csv'
    wordlimit = 14000
    visoutput = 'Negative_LDA_Visualization.html'
elif num == 2:
    csv = 'neutral.csv'
    wordlimit = 3500
    visoutput = 'Neutral_LDA_Visualization.html'
elif num == 3:
    csv = 'positive.csv'
    wordlimit = 20000
    visoutput = 'Positive_LDA_Visualization.html'
else:
    print('Invalid input! Please enter a valid argument: ')
    print('main.py <1, 2 or 3> | 1 = negative.csv, 2 = neutral.csv, 3. positive.csv')
    sys.exit()

# Chapter 1: Initialize data to be 'cleaned'
import re
import pandas as pd
df = pd.read_csv(csv)
df.head()

# Chapter 2: Clean Data
'''
Removing Non-English words is crucial in this task of topic modeling
as the task requires analysis of each topic within each emotion category
For the simplicity of the task, non-English words are removed. SpaCy,
nltk, re and string are used for this chapter.
'''

# Initiate nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initiate cleaning stage.
import string
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Stopwords to remove 'useless' words
sWords = stopwords.words('english')
sWords.extend(['got', 'say', 'use', 'from', 'gon', 'na', 'wa', 'nt', 'gt', 'to', 'also', 'that', 'this', 'the'])
setStopWords = set(sWords)
puncExclude = set(string.punctuation)
engWords = set(nltk.corpus.words.words())

#
lemma = WordNetLemmatizer()

# Load Spacy Languge Support
nlp = spacy.load('en_core_web_sm')

# Assign English Language detector to appropriate function
def get_lang_detector(nlp, name):
    return LanguageDetector()
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe("language_detector", last=True)

# Initialise Data Cleaning

# Stage 1: Remove URLs, Newlines, Numbers, Usernames, Punctuations
# Remove URLs
df['cleantext'] = df['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

# Remove Newlines
df['cleantext'] = df['cleantext'].replace(r'\n',' ', regex=True)
df['cleantext'] = df['cleantext'].replace(u'\xa0', u' ', regex=True) 

# Remove Numbers
df['cleantext'] = df['cleantext'].str.replace('\d+', '',regex=True)

# Remove Usernames
df['cleantext']= df['cleantext'].str.replace('(@\w+.*?)',"",regex=True)

# Remove Punctuations
df['cleantext'] = df['cleantext'].str.replace('[^\w\s]','',regex=True)

# Input all text values into a list.
# Each text value will be referred to in this code as a document.
data = df.cleantext.values.tolist()

# Stage 2: Advanced Cleaning - Remove Stopwords and Non-English sentences & Normalize Corpus
def clean(doc):
    wordTokens = word_tokenize(doc)

    # Remove Non-English words
    removeNonEng = " ".join(w for w in wordTokens if w.lower() in engWords or not w.isalpha())
    
    # Normalize Corpus
    normaldata = " ".join(lemma.lemmatize(word) for word in removeNonEng.split())

    # Remove Stopwords
    removesw = " ".join([x for x in normaldata.split() if not x.lower() in setStopWords])
    
    # Final Removal of Non-English Sentences
    data1 = nlp(removesw)
    finalData = ""
    for sent in data1.sents:
        if (sent._.language)['language'] == 'en':
            finalData += str(sent)
    finalData = finalData.strip()

    return finalData


# Data cleaning completed.
cleanData = [clean(doc).split() for doc in data]

# Chapter 3: Prepare Document-Term Matrix
'''
All of the rows in the 'text' column, when combined
together, is known as the corpus. Converting this 
into matrix representation allows LDA model to
easily find repeating term patterns in the DT matrix.
For this, we'll use gensim.
'''

import gensim
from gensim import corpora

# Create term dictionary for corpus
corpdict = corpora.Dictionary(cleanData)

# Convert list of documents into DTM using above dictionary
docTermMatrix = [corpdict.doc2bow(doc) for doc in cleanData]

# Creating LDA model using gensim
Lda = gensim.models.ldamodel.LdaModel

# Run and Train LDA model on the DTM
model = Lda(docTermMatrix, num_topics=10, id2word=corpdict, passes=50)

# Chapter 4: Visualizing each Topic into a Wordcloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

# Coloring scheme for each topic
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

cloud = WordCloud(stopwords=setStopWords,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=25,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = model.show_topics(formatted=False)

# Plot wordclouds in a 2 x 5 format
fig, axes = plt.subplots(2, 5, figsize=(5,5), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topicWords = dict(topics[i][1])
    cloud.generate_from_frequencies(topicWords, max_font_size=300)
    plt.gca().imshow(cloud, interpolation='bilinear')
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

# Chapter 5: Visualizing Topics via word counts of keywords
from collections import Counter
topics = model.show_topics(formatted=False)
data_flat = [w for w_list in cleanData for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Wordcount and Weights of Topic Keywords in 2 x 5 format
fig, axes = plt.subplots(2, 5, figsize=(10,9), sharey=True)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.3, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.950); ax.set_ylim(0, wordlimit)
    ax.set_title('Topic: ' + str(i+1), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.subplots_adjust(left = 0.045, right = 0.971, wspace = 0.129, hspace = 0.267)
plt.show()

# Chapter 6: Topic Visualization
'''
In order to properly present our data using our model,
pyLDAvis provides a good and interactive way of seeing each topic
keywords and maps each topic with different graphs.
'''
import pyLDAvis.gensim_models
import pyLDAvis

# Visualize the topics
visualisation = pyLDAvis.gensim_models.prepare(model, docTermMatrix, corpdict)
pyLDAvis.save_html(visualisation, visoutput)

