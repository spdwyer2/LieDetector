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

#This is a broken section. I was trying to binarize my outcomes so that I could use cross validation after training my model
#from sklearn.preprocessing import label_binarize
#df['rulings'] = label_binarize(df['rulings'], classes=[0, 1, 2, 3, 4, 5, 6])

from sklearn.svm import SVC
import seaborn
clf2 = SVC(C=10,gamma=1)
clf2.fit(tf_idf, df['rulings'])
print clf2.score(tf_idf,df['rulings'])
rulings_predictions2 = clf2.predict(tf_idf)
cm = confusion_matrix(df['rulings'],rulings_predictions2)
print cm
print seaborn.heatmap(cm)


#This is a broken section. As mentioned above, this is where I was attempting to use AUC to cross validate my model.
#from sklearn import cross_validation
#scores = cross_validation.cross_val_score(clf2, tf_idf, df['rulings'], cv=10,scoring='roc_auc')

#print the average cross-validated Area Under the Curve for that model
#print 'Average AUC %f' % scores.mean()


#This is a list of just text from pants-fire and false rulings. This will be used for topic modeling.
false_rulings = df.loc[df['rulings'].isin(['pants-fire','false'])]['text']

#Begin topic modeling 
from sklearn.decomposition import LatentDirichletAllocation
count_vect = CountVectorizer()
new_counts = count_vect.fit_transform(false_rulings)

#Fit a model to the above matrix with 10 possible topics
n_topics = 10
lda = LatentDirichletAllocation(n_topics=n_topics,max_iter=10)
lda.fit(new_counts)

#Find the 20 most important words within the most important topics
n_top_words = 20
import operator
vocabulary = count_vect.get_feature_names()

for i in range(n_topics):
    best_words_indexes = lda.components_[i].argsort()[:-n_top_words - 1:-1]
    best_words = " ".join([vocabulary[i] for i in best_words_indexes])
    print best_words


#Build a matrix with the topics as well as each observation's score for that topic
n_topics = 10
topic_assignments = lda.transform(new_counts)
topic_names = ["topic%s" % i for i in range(1,n_topics+1)]
topic_assignments = pd.DataFrame(topic_assignments,columns=["topic1","topic2","topic3","topic4","topic5","topic6","topic7","topic8","topic9","topic10"], index=df.index)

#This is a broken section. I was trying to build the topic scores into the orginal matrix so that I could run a 
#random forest model on the whole thing and get importance scores
#topic_assignments.index = df[df['rulings'].isin(['pants-fire','false'])].index
#df = df.append(topic_assignments)
#df_final = pd.concat([df, topic_assignments], axis=1, join_axes=[df.index])
#df.to_csv('scotttry8.csv')







