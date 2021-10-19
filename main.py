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
    import gensim, spacy, warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # spacy for lemmatization
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    # Plotting tools
    import pyLDAvis.gensim_models
    import matplotlib.pyplot as plt
    
    # NLTK Stopwords
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['got', 'say', 'use', 'from', 'gt', 'to', 'also', 'that', 'this', 'the'])

    # Chapter 2: Importing Sample Data
    # Import Dataset
    df = pd.read_csv('redditandtweets.csv')
    df.head()

    # Chapter 3: Cleaning and Tokenizing Sentences
    # Convert To List
    data = df.text.values.tolist()

    # Remove Punctuations and Unnecessary characters
    def sent_to_words(sentences):
        for sent in sentences:
            # Remove Newline characters
            sent = re.sub('\s+', ' ', sent)

            # Remove distracting single quotes
            sent = re.sub("\'", "", sent)

            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sent), deacc=True))

    data_words = list(sent_to_words(data))

    # Chapter 4: Building Bigram, Tigram Models and Lemmatize
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 50)
    trigram = gensim.models.Phrases(bigram[data_words], threshold = 50)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        # Remove Stopwords, Form Bigrams, Trigrams and Lemmatization
        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
        texts_out = []

        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
        return texts_out

    data_final = process_words(data_words) # Processed Text Data

    # Chapter 5: Building the Topic Model
    # Create Dictionary
    corpdict = corpora.Dictionary(data_final)

    # Create Corpus
    texts = data_final

    # Term Document Frequency
    corpus = [corpdict.doc2bow(text) for text in texts]

    # Build LDA Model
    model = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = corpdict, num_topics=20, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    doc_lda = model[corpus]

    '''# Compute Perplexity
    print('\nPerplexity: ', model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model, texts=data_lemmatized, dictionary=corpdict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    pprint(model.print_topics())'''

    def visualiseTopics(ldamodel=None, corpus=corpus, texts = data):
        # Initialize Output
        sentdf = pd.DataFrame()

        # Get main topic in each document
        for i, rowlist in enumerate(ldamodel[corpus]):
            row = rowlist[0] if ldamodel.per_word_topics else rowlist
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            for j, (topicnum, proptopic) in enumerate(row):
                    if j == 0:
                        wordprop = ldamodel.show_topic(topicnum)
                        topickeywords = ", ".join([word for word, prop in wordprop])
                        sentdf = sentdf.append(pd.Series([int(topicnum), round(proptopic,4), topickeywords]), ignore_index=True)
                    else:
                        break
        sentdf.columns = ['Dominant_Topic', 'Percentage_Contribution', 'Topic_Keywords']
        # Add original text to the end of the output
        contents = pd.Series(texts)
        sentdf = pd.concat([sentdf, contents], axis=1)
        return(sentdf)

    df_topic_sents_keywords = visualiseTopics(ldamodel=model, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Percentage_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(20)

    # Create Wordcloud
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.colors as mcolors

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    


