'''
Author: Scott Dwyer
First created: August 13
Questions? spdwyer2@gmail.com
'''

#arbitrary change


import pandas as pd
import requests
from bs4 import BeautifulSoup, BeautifulStoneSoup
import re 

#pull Trump-specific statements from politifact. The "n" determines how many statements are pulled. NOTE: it's left at 1 right now just for simplicity's sake. I plan on upping it to 180 once I get the formatting right.
statements = requests.get("http://www.politifact.com/api/statements/truth-o-meter/people/donald-trump/xml/?n=200")

#Encode the xml in utf-8 because the internet is terrible
statements = statements.text.encode("utf-8")

#the lxml library allows us to parse the xml formmated file we'll received after we make our request
import lxml.etree as ET

#save the text that was returned into a string
doc = ET.fromstring(statements)

doc.findall('resource/ruling/ruling_slug')
#These are the lists for the above attributes (truth, date, subject, etc.) that I'll append with individual comments using the below for loops
rulings = []
names = []
dates = []
subjects = []
statements = []

#a set is a list that only allows unique elements
unique_subjects = set()

#These are all my for loops - they populate the above lists. They strip the XML tags using get_text()
for element in doc.findall('resource/ruling/ruling_slug'):
	ruling_result = element.text
	rulings.append(ruling_result)

for element in doc.findall('resource/ruling_date'):
	ruling_date = element.text
	dates.append(ruling_date)

for element in doc.findall('resource'):
	subject_lst=[]
	for element2 in element.findall("subject/resource/subject_slug"):
		subject_lst.append(element2.text)
		unique_subjects.add(element2.text)
	ruling_subject = " ".join(subject_lst)
	subjects.append(ruling_subject)

for element in doc.findall('resource/speaker/name_slug'):
	ruling_names = element.text
	names.append(ruling_names)

for element in doc.findall('resource/statement'):
	ruling_statement = element.text
	statements.append(ruling_statement)

data = {}
data['rulings'] = rulings
data['names'] = names
data['dates'] = dates
data['subjects'] = subjects
data['statements'] = statements

df = pd.DataFrame.from_dict(data)

#clean up the statements
df['statements'] = df['statements'].apply(lambda x: re.sub('<[^>]*>|','',x).strip())
df['statements'] = df['statements'].apply(lambda x: re.sub('&#39;',"'",x).strip())


#Make dummy codes for the subjects
for subject in unique_subjects:
	df[subject] = df.subjects.apply(lambda x: 1 if subject in x else 0 )

#This function looks for statements that have a Trump quote and splits those statements on the quote
def get_quote(statement):
	if "&quot" in statement: 
		return statement.split("&quot")[1]
	return None 

#Here I'm applying the get_quote function to the statements column and rewriting results to a new column
df['quote'] = df['statements'].apply(get_quote)
df['text'] = df['quote']

###
#Begin NLP Section
###

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# Remove stop words
def remove_stopwords(content):
    if content == None:
		return None 
    cleaned = filter(lambda x: x not in ENGLISH_STOP_WORDS,content.split())
    return ' '.join(cleaned)

#Apply the function to every row
df['text'] = df['text'].apply(remove_stopwords)

#remove punctuation and numbers
import string
def remove_punctuation(content):
	if content == None:
		return None
	return filter(lambda x: x in string.ascii_letters+" ",content)

df['text'] = df['text'].apply(remove_punctuation)


# Reducing strings to their stemmed words
import porterstemmer
stemmer = porterstemmer.PorterStemmer()
def stem_words(content):
	print content
	if str(content) == "None":
		return None
	stemmed_words = [stemmer.stem(word, 0,len(word)-1) for word in content.split()]
	return " ".join(stemmed_words)

df['text'] = df['text'].apply(stem_words)

#Storing the cleaned quote text as a variable
df['text']= df['text'].fillna("")

#Training a classifier using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['text'])

#Creating tf-idf scores for each comment
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=True).fit(counts)
tf_idf = tf_transformer.transform(counts)

#Convert from sparse form to normal matrix
tf_idf.todense()

#See how well the tf-idf scores predict rulings
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(tf_idf, df['rulings'])
print clf.score(tf_idf,df['rulings'])

#Check out the confusion matrix
from sklearn.metrics import confusion_matrix
rulings_predictions = clf.predict(tf_idf)
cm = confusion_matrix(rulings,rulings_predictions)
print cm


from sklearn.svm import SVC
import seaborn
clf = SVC(C=10,gamma=1)
clf.fit(tf_idf, df['rulings'])
print clf.score(tf_idf,df['rulings'])
rulings_predictions2 = clf.predict(tf_idf)
cm = confusion_matrix(df['rulings'],rulings_predictions2)
print cm
seaborn.heatmap(cm)





df.to_csv('scotttry6.csv')



'''

Ivan advice:
Create TFIDF score for every single statement
These scores are for words
Then have topic scores for every observation

2 big ways to encode text:
content: sentiment, topic (religion? politics? Money?), lots of adjectives, etc.?
style: are you using diverse amount of words? How readable is it? Flesh concade readability score. Readability = length of words per sentence.
Count of commas!!!!! Normalize by sentences. Liars qualify sentences more. 
topic: first you have to wrap a count vectorizer (sklearn) what it does is transforms your text into a matrix where rows are statements and columns are the presence
of certain words. From there you can pass this is on to LDA and that will look at coocurrences of certain words. Tell it how many topics you think there are. 

Scott notes to self:
Look at distribution of rulings and use clf class prior to weight the probability of any decision
Try 


'''

