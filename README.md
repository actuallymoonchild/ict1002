# ict1002

Task 3: Analyze the topics within each emotion categorizes using topic modeling or detection. 

How to Run:

1. Open CMD
2. Navigate to folder containing main.py
3. Type in CMD: main.py <1, 2 or 3> **(IMPORTANT)**
> 1 signifies that the code will present the data based on negative.csv.\
> 2 signifies that the code will present the data based on neutral.csv.\
> 3 signifies that the code will present the data based on positive.csv.
4. Wordcloud Output will be displayed first. You can choose to export the data to .png by clicking the save icon on the bottom left.
5. Wordcount and Weightage Graph per Topic will then be displayed. You can choose to export the data to .png by clicking the save icon on the bottom left.
6. LDA Visualization will automatically save a .html file in the current folder. The name of the html output is dependent on the argument input.

Libraries Used:\
sys: to take in user input as arguments\
re: used to clean data by removing URLs\
pandas: used to read csv files and store them into a dataframe\
nltk: used to clean data by removing stopwords and natural language processing\
string: used to clean data by removing punctutions \
spacy: used to clean non-english data with the help of nltk\
gensim: used to prepare document-term matrix and LDA model for topic modeling\
matplotlib: used to plot data into a proper structured and clear format\
wordcloud: similar to matplotlib but into a wordcloud instead\
pyldavis: used to visualize topics in an interactive html page

