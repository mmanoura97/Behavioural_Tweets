# utilities
import re # a built-in package which can be used to work (search, split etc.) with Regular Expressions
import pickle # module used for serializing and de-serializing python object structures (eg list --> bytes (0,1))
import numpy as np # python library used for working with arrays
import pandas as pd # python library pandas is used to analyze data

# plotting
import seaborn as sns # seaborn is a library that uses Matplotlib underneath to plot graphs, it is used to visualize random distributions.
from wordcloud import WordCloud # wordcloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance
import matplotlib.pyplot as plt # matplotlib is a python library used to create 2D graphs and plots by using python scripts-module pyplot makes things easy for plotting
                                # by providing feature to control line styles, font properties, formatting axes etc.


# nltk # nltk is a leading platform for building Python programs to work with human language data
from nltk.stem import WordNetLemmatizer # package used to remove morphological affixes(grammatical role, tense, derivational morphology etc) from words, leaving only the word stem
import nltk # collection of python libraries  for identifying and tag parts of speech found in the text of natural language
nltk.download('omw-1.4') # omw = Open Multilingual Wordnet (wordnet = an English dictionary) that share a common function with nltk, but produce different output types:

# sklearn # sklearn (= Scikit-learn) is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning
          # and statistical modeling including classification, regression, clustering and dimensionality reduction
from sklearn.svm import LinearSVC # Linear SVC is an algorithm that attempts to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data.
from sklearn.naive_bayes import BernoulliNB # Bernoulli Naive Bayes is one of the variants of the Naive Bayes algorithm in machine learning. It is based on the Bernoulli Distribution
                                            # and accepts only binary values, i.e., 0 or 1. If the features of the dataset are binary, then we can assume
                                            # that Bernoulli Naive Bayes is the algorithm to be used.
from sklearn.linear_model import LogisticRegression # Logistic regression is a fundamental classification technique. It predicts categorical outcomes, unlike linear regression
                                                    # that predicts a continuous outcome

from sklearn.model_selection import train_test_split # method used to estimate the performance of machine learning algorithms that are applicable for prediction-based Algorithms/Applications.
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features.
                                                            # TF-IDF stands for term frequency-inverse document frequency and it is a measure that can quantify the importance or relevance
                                                            # of string representations (words, phrases, lemmas, etc) in a document amongst a collection of documents
from sklearn.metrics import confusion_matrix, classification_report # a confusion matrix is such that C i , j is equal to the number of observations known to be in group
                                                                    # and predicted to be in group. Thus in binary classification, the count of true negatives is C 0 , 0 ,
                                                                    # false negatives is C 1 , 0 , true positives is C 1 , 1 and false positives is C 0 , 1 .
                                                                    # a performance evaluation metric in machine learning which is used to show the precision, recall, F1 Score,
                                                                    # and support score of your trained classification model.
# Importing the dataset (csv file derrived from Kaggle)
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
file_name = r'C:\Users\mmanoura\OneDrive - Deloitte (O365D)\Desktop\Github\ML projects\Behavioural Tweets\Dataset\training.1600000.processed.noemoticon.csv'
dataset = pd.read_csv(file_name, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Removing the unnecessary columns
dataset = dataset[['sentiment','text']]
# Transposition of the values in column 1 with column 4 and vice versa to ease understanding.
dataset['sentiment'] = dataset['sentiment'].replace(4,1)


# Plotting the distribution for dataset.
# with groupby() I classify the data according to the sentiment
# with count() I count how many tweets I have in each sentiment
# with plot() create the diagram: number of tweets-sentiment
# kind: determines whether the diagram will be histogram (hist), bar plot (bar), pie plot (pie), etc.
# we can also define title and legend
ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False) #figure 1: Number of tweets-Sentiment
# we can also define the labels of each sentiment and if we want we can rotate the label
ax.set_xticklabels(['Negative','Positive'], rotation=0)
#creating the plot: Number of tweets-Sentiment
#plt.show()

# Storing data in lists: in the list text I store all the data from the column text of the dataset
# and in the list sentiment I store all the data from the column sentiment of the dataset
text, sentiment = list(dataset['text']), list(dataset['sentiment'])


# Defining dictionary containing with ---keys: all emojis
                                    # ---values: their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Defining set containing all stopwords in english.
# The stopwords are a list of words that are very, very common but don’t provide useful information for most text analysis procedures.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'day' , 'did', 'do',
             'does', 'don' , 'dont' , 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'go' , 'going' , 'got', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'im' , 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','much', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 'quot' , 'quote' ,
             're', 's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'time' , 'to', 'today' ,  'too', 'twitter' , 'URL' , 'under', 'until', 'up',
             'USER' , 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
             'which','while', 'who', 'whom', 'why', 'will', 'with', 'won', 'y', 'you',
             "youd","youll", "youre", "youve", 'your', 'yours', 'yourself', 'yourselves']

# Creating function preprocess with argument: textdata
def preprocess(textdata):
    # initializing the list processedText
    processedText = []

    # create Lemmatizer and Stemmer.
    # Lemmatization is the process of converting a word to its base form (root).
    # The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes
    # the last few characters, often leading to incorrect meanings and spelling errors.

    # In order to lemmatize, i need to create an instance of the WordNetLemmatizer() and call lemmatize() function on a single word.
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    # A Regular Expression (or Regex) is a pattern (or filter) that describes a set of strings that matches the pattern. In other words, a regex accepts a certain set of strings
    # and rejects the rest.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    # in textdata argument (that we will match with text list, the list containing the tweets), for every single tweet, we convert all characters to lower case
    for tweet in textdata:
        tweet = tweet.lower()
        # The sub() function replaces the matches with the text of our choice - here we replace all URls with 'URL'
        tweet = re.sub(urlPattern, ' URL', tweet)
        # Replace all emojis - here i replace each emoji(key) with its match(value) from emojis dictionary
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            # The sub() function replaces the matches with the text of our choice - here we replace all @USERNAME with 'USER'
        tweet = re.sub(userPattern, ' USER', tweet)
        # Replace all non alphabets with " ".
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        # initializing the varaible tweetwords
        tweetwords = ''
        # for each word in each tweet i am checking for stopwords
        for word in tweet.split():
            # Checking if the word is a stopword.
            if word not in stopwordlist:
                if len(word) > 1: #len() function returns the number of characters in each word
                    # Lemmatizing the word withthe instance of the WordNetLemmatizer(), i created
                    word = wordLemm.lemmatize(word)
                    # in tweetwords variable i add all the words for each tweet
                    tweetwords += (word + ' ')
        # finally on processedText list i add all the processed tweets
        processedText.append(tweetwords)

    return processedText


import time
# function time.time() is used to return the current time of the system
t = time.time()

# i am calling preprocess function to process the text i will use, and i store it on list processedText
processedtext = preprocess(text)
print(f'Text Preprocessing complete.')
# i subtract from the current time, the time the execution started to find out how long preprocessing lasted.
print(f'Time Taken: {round(time.time()-t)} seconds')

#i store the first 800000 processed tweets, thus negative tweets (from list processed text) on list data_neg
data_neg = processedtext[:800000]

#creating negative wordcloud
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg)) #with join i create a string from the data_neg, seperated with " "
#The figure module is used to control  all the plot elements-figsize parameter is the Figure dimension (width, height) in inches
plt.figure(figsize = (10,10)) #figure 2: negative wordcloud
plt.imshow(wc)

#i store the last 800000 processed tweets, thus positive tweets (from list processed text) on list data_neg
data_pos = processedtext[800000:]

#creating positive wordcloud
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800,
              collocations=False).generate(" ".join(data_pos)) # with join i create a string from the data_pos,
                                                               # seperated with " " (data_pos without join is a list of strings, not a string).

#The figure module is used to control  all the plot elements-figsize parameter is the Figure dimension (width, height) in inches
plt.figure(figsize = (10,10))
plt.imshow(wc) #figure 3: positive wordcloud

#printing the 3 figures
plt.show()

# now we will split the data: ---Training Data: The dataset upon which the model would be trained on. Contains 95% data.
                            # ---Test Data: The dataset upon which the model would be tested against. Contains 5% data.
#X: processed text (from each tweet) and y: sentiment (from each tweet)
#with test_size i define the percentage of the test data - the rest will be train data
X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment,
                                                    test_size = 0.05, random_state = 0)
print(f'Data Split done.')

# TF-IDF Vectoriser converts a collection of raw documents to a matrix of TF-IDF features. The Vectoriser is usually trained on only the X_train dataset.
# TF-IDF Vectoriser vectorizes a corpos and prepares it to be input into an estimator (sklearn doesn't accept 'text form' features, thus i need to transform them using TfidfVectorizer.
# TF-IDF Vectoriser indicates what the importance of the word is in order to understand the dataset.
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000) # ngram_range: n-gram is just a string of n words in a row. With parameter ngram_range=(a,b)
                                                                     # where a is the minimum and b is the maximum allows us to set the size of ngrams we want to include in our features.
                                                                     # max_features: specifies the number of features to consider (ordered by features that occur most frequently).
# training the vectoriser
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names_out())) # Get output feature names' for transformation length

# I fit the training set and transform the validation set.
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformed.')

print("flag")
#categories = ['Negative', 'Positive']
#print(categories)
def model_Evaluate(model):
    # Predict values for Test dataset
    # Given a trained model, we predict the label of a new set of data.
    # This method accepts one argument, the new data X_test (e.g. model.predict(X_test)), and returns the learned label for each object in the array.
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    # Classification report is a performance evaluation metric in machine learning which is used to show the precision, recall, F1 Score
    # and support score of your trained classification model.
    # Precision is defined as the ratio of true positives to the sum of true and false positives.
    # Recall is defined as the ratio of true positives to the sum of true positives and false negatives.
    # The F1 is the weighted harmonic mean of precision and recall. The closer the value of the F1 score is to 1.0, the better the expected performance of the model is.
    # Support is the number of actual occurrences of the class in the dataset. It doesn’t vary between models, it just diagnoses the performance evaluation process.
    print(classification_report(y_test, y_pred))
    print("classification report")

    # Compute and plot the Confusion matrix
    # The confusion matrix is often used in machine learning to compute the accuracy of a classification algorithm.
    # It measures the quality of predictions from a classification model by looking at how many predictions are True and how many are False.
    # Specifically, it computes: True positives (TP), False positives (FP), True negatives (TN), False negatives (FN).
    cf_matrix = confusion_matrix(y_test, y_pred)

    """
    # Creating list categories, that contains Negative, Positive
    categories = ['Negative', 'Positive']
    # Creating list group_names, that contains True Negative, False Positive, False Negative, True Positive
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)

    BNBmodel = BernoulliNB(alpha=2)
    BNBmodel.fit(X_train, y_train)
    model_Evaluate(BNBmodel)

    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train, y_train)
    model_Evaluate(SVCmodel)

    LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    LRmodel.fit(X_train, y_train)
    model_Evaluate(LRmodel)

    file = open('vectoriser-ngram-(1,2).pickle', 'wb')
    pickle.dump(vectoriser, file)
    file.close()

    file = open('Sentiment-LR.pickle', 'wb')
    pickle.dump(LRmodel, file)
    file.close()

    file = open('Sentiment-BNB.pickle', 'wb')
    pickle.dump(BNBmodel, file)
    file.close()

    def load_models():
        '''
        Replace '..path/' by the path of the saved models.
        '''

        # Load the vectoriser.
        file = open('..path/vectoriser-ngram-(1,2).pickle', 'rb')
        vectoriser = pickle.load(file)
        file.close()
        # Load the LR Model.
        file = open('..path/Sentiment-LRv1.pickle', 'rb')
        LRmodel = pickle.load(file)
        file.close()

        return vectoriser, LRmodel

    def predict(vectoriser, model, text):
        # Predict the sentiment
        textdata = vectoriser.transform(preprocess(text))
        sentiment = model.predict(textdata)

        # Make a list of text with sentiment.
        data = []
        for text, pred in zip(text, sentiment):
            data.append((text, pred))

        # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(data, columns=['text', 'sentiment'])
        df = df.replace([0, 1], ["Negative", "Positive"])
        return df

    if __name__ == "__main__":
        # Loading the models.
        # vectoriser, LRmodel = load_models()

        # Text to classify should be in a list.
        text = ["I hate twitter",
                "May the Force be with you.",
                "Mr. Stark, I don't feel so good"]

        df = predict(vectoriser, LRmodel, text)
        print(df.head())

        print('finish')"""