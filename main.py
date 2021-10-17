'''
This programme is to analyze topics of discussion
within each emotion category using topic modeling.
Each .csv file has been provided based on the sentiment
analysis of discussions on Twitter and Reddit.

Columns are:
Row Number, Text, Date, Link
'''

if __name__ == '__main__':

    import nltk; nltk.download('stopwords', quiet=True)

    import re
    import numpy as np
    import pandas as pd
    from pprint import pprint

    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # spacy for lemmatization
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    # Plotting tools
    import pyLDAvis
    import pyLDAvis.gensim_models
    import matplotlib.pyplot as plt

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # NLTK Stopwords
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    # Import Dataset
    df = pd.read_csv('redditandtweets.csv')
    df.head()

    # Convert To List
    data = df.text.values.tolist()

    # Remove Newline characters
    data = [re.sub('\s+', ' ', text) for text in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", text) for text in data]

    # Remove Punctuations and Unnecessary characters
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and tigram models
    bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 50)
    trigram = gensim.models.Phrases(bigram[data_words], threshold = 50)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams)

    # Create Dictionary
    corpdict = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [corpdict.doc2bow(text) for text in texts]

    # Build LDA Model
    model = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = corpdict, num_topics=100, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    doc_lda = model[corpus]

    # Compute Perplexity
    print('\nPerplexity: ', model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model, texts=data_lemmatized, dictionary=corpdict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    vis = pyLDAvis.gensim_models.prepare(model, corpus, corpdict)
    pyLDAvis.save_html(vis, 'visualization.html')