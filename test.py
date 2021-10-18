'''
This programme is to analyze topics of discussion
within each emotion category using topic modeling.
Each .csv file has been provided based on the sentiment
analysis of discussions on Twitter and Reddit.

Columns are:
Row Number, Text, Date, Link
'''

# Chapter 1: Initialize data to be 'cleaned'
import pandas as pd
df = pd.read_csv('redditandtweets.csv')
df.head()

# Input all text values into a list.
# Each text value will be referred to in this code as a document.
data = df.text.values.tolist()

# Initiate cleaning stage.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Stopwords to remove 'useless' words
sWords = stopwords.words('english')
sWords.extend(['got', 'say', 'use', 'from', 'gt', 'to', 'also', 'that', 'this', 'the'])
stop_words = set(sWords)
puncExclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Chapter 2: Clean Data
def clean(doc):
    word_tokens = word_tokenize(doc)
    # Remove Stopwords
    removesw = " ".join([x for x in word_tokens if not x.lower() in stop_words])

    # Remove Punctuations
    removepunc = ''.join(char for char in removesw if char not in puncExclude)

    # Normalize Corpus
    normaldata = " ".join(lemma.lemmatize(word) for word in removepunc.split())
    
    return normaldata

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

# Result here
print(model.print_topics(num_topics=50, num_words=20))
