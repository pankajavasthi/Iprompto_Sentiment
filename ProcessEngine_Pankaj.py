
import os
import sys
import time
import csv
import pandas as pd
import numpy as np
import itertools
#import matplotlib.pyplot as plt
import pyodbc
import sqlalchemy
import logging
from sqlalchemy import create_engine
from sqlalchemy import sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric  
from sqlalchemy.orm import sessionmaker
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet
from nltk.corpus import webtext
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.stem import WordNetLemmatizer
from nltk.chunk.regexp import ChunkString, ChunkRule, ChinkRule, RegexpParser
from nltk.tree import Tree
from nltk.tag import untag
from featx import bag_of_words
"""

#from tabulate import tabulate
#from langdetect import detect

"""
from string import punctuation
from NBClassify import nb_classifier
from collections import Counter
#from wordcloud import WordCloud
#from scipy.misc import imread
from functools import reduce
from functools import partial
#rake related
import rake
#scipy related
import numpy as np
import re
from scipy import stats
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
#for joblib stuff
from sklearn.externals import joblib #added by Rajesh_Rajamani on 27-May-2016

# For Personality by Ajay
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing


# For NER tagging
from itertools import groupby
java_path = "C:/Program Files/Java/jdk1.8.0_91/bin/java.exe"

os.environ['JAVAHOME']=java_path



from nltk.tag import StanfordNERTagger
st7 = StanfordNERTagger('D:/Text Tool/stanford-ner-2014-06-16/classifiers/english.muc.7class.distsim.crf.ser.gz','D:/Text Tool/stanford-ner-2014-06-16.zip')
st3 = StanfordNERTagger('D:/Text Tool/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','D:/Text Tool/stanford-ner-2014-06-16.zip')
st4 = StanfordNERTagger('D:/Text Tool/stanford-ner-2014-06-16/classifiers/english.conll.4class.distsim.crf.ser.gz','D:/Text Tool/stanford-ner-2014-06-16.zip')



#Directories
os.chdir("D:\\Text Tool\\Python Desktop")

smartstoppath = "D:\\Text Tool\\Python Desktop"

#files
stoppath = smartstoppath +"\\SmartStoplist.txt"
    
# Used to extract important words for root cause(not using in display)    
rake_object = rake.Rake(stoppath)

#stopword pattern
stopwordpattern = rake.build_stop_word_regex(stoppath)    



#Sys Arguments - Apart from File name , Batch ID to be passed from the web application
#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))
# Logging variables
def LogFileHandler(batchid):
	# create a file handler
	handler = logging.FileHandler(batchid+'.log')
	handler.setLevel(logging.INFO)
	
	# create a logging format
	formatter = logging.Formatter('%(asctime)s , %(message)s , %(name)s , %(levelname)s ', datefmt='%Y-%m-%d %H:%M:%S' )
	handler.setFormatter(formatter)

	# add the handlers to the logger
	logger.addHandler(handler)

	
#Classes
Base = declarative_base()

#Declaration of the class in order to write into the database. This structure is standard and should align with SQLAlchemy's doc.
class Current(Base):
    __tablename__ = 'tableName'


    id = Column(Integer, primary_key=True)
    Date = Column(String(500))
    Type = Column(String(500))
    Value = Column(Numeric())
    
    def __repr__(self):
        return "(id='%s', Date='%s', Type='%s', Value='%s')" % (self.id, self.Date, self.Type, self.Value)
    
#Functions

#Rake Functions
#function for split sentences
def getsplitsentences(text):
    sentenceList = rake.split_sentences(text)
    return sentenceList
    

#function for phraselist
def getphraselist(sentenceList):
    phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern)
    return phraseList
    

#function for # generate candidate keyword scores
def getcandidatescores(phraseList):
    wordscores = rake.calculate_word_scores(phraseList)
    keywordcandidates = rake.generate_candidate_keyword_scores(phraseList, wordscores)
    return keywordcandidates

stemmer = SnowballStemmer('english')
#stemming scipy functions
def stemming(string):
    s = ""
    string = string.split()
    for e in string:
        s = s + stemmer.stem(e)+ " "
    return s

def listtostr(listring):
    s = ""
    for e in listring:
        s = s + e + " , "
    return s

	

def get_dict(cursor,sql):
    
    cursor.execute(sql)
    #declare a dictionary
    schema =  {}
    #get column name
    colname = "tokenword"
        
    for row in cursor.fetchall():
        schema.setdefault(colname, []).append(row[-1])
      
    return (schema) 
    #close cursor

def connectionstring():
    connectionstring='DRIVER={SQL Server};SERVER=IGSDLF-NB-0040\SQLEXPRESS2012;DATABASE=TXTIntelligence;UID=sa;PWD=India@123'
    return connectionstring
    

def database_connect():
    #returns cursor
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=IGSDLF-NB-0040\SQLEXPRESS2012;DATABASE=TXTIntelligence;UID=sa;PWD=India@123',autocommit=True)
    cursor = cnxn.cursor()
    return cursor

def dbcon():
    #returns connection
    cnxn = pyodbc.connect(connectionstring())
    return cnxn

def UpdateBatchFlag(FileUMID, flag): 
    #print("reached Here")
    cursor = database_connect() # Connect to database        
    sql_query = "exec Proc_EDI_Update_PythonStatus '%s', '%s'" % (FileUMID, flag)    
    #print(sql_query)
    cursor.execute(sql_query)
    cursor.close

def PrepSQLForRecords(batchid): 
    #print("reached Here")
    cursor = database_connect() # Connect to database        
    sql_query = "exec usp_InsertAnalysisColumnsSelect '%s'" % batchid    
    #print(sql_query)
    #cursor.execute(sql_query)
    cursor.close
	
  
def get_list(cursor,sql):
    
    cursor.execute(sql)
    #declare a dictionary
    schema =  []
    
        
    rows = [x[0] for x in cursor.fetchall()]    
    for row in rows:
        schema.append (row)
      
    #print (schema)
    #print("/n")
    return (schema) 
  

def WordBucket(text,dictionary):
    
    cdict = {} # dictionary
    #dictionary = Pos_words
    cdict = dictionary # assign the dictionary name received ( this dictionary should be already populated)
    WordList=[] 
       
    for eachitem in text:
        #print(eachitem)
        #print("/n")
        for item in cdict.values():
            
            if eachitem in item:
                WordList.append(str(eachitem))
    
    return WordList


def print_elapsed_time(timestamp,step):
    elapsed_time = time.clock() - timestamp
    print ("Step Completed %s" %step)            
    print ("Time elapsed: {} seconds".format(elapsed_time))
    
    return time.clock()
    
    
def SentimentScore(text):


    scr1 = 0
    N = len(text)
        
    for i in range(N):
        if text[i] in PosWords.values():            
            scr1 = scr1 + 1
            if i >1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 + 1
                if text[i-1] in NotWords:
                    scr1 = scr1 - 2
        elif text[i] in NegWords:
            scr1 = scr1 - 1
            if text[i] in NegHighWords:
                scr1=scr1-2
            if i > 1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 - 1
                if text[i-1] in NotWords:
                    scr1 = scr1 + 2
    return scr1    

'''-----------------------------------------------------------------------------------------'''
'''----------- Cleaning Level 1 & 2 --------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
#Word Sets
cachedStopWords = stopwords.words("english")

web_words = ['https:','"','""','",','""!' ,'-','+','=','_','~','^','^','?','.','..','...','....','(?)', ',', '&','%','!','@','$','!!','!!!','|','||','|||','/','\\','\\.', '+','(',')','<','>','><','u','h', 'www','img','border','color','style','padding','table','font','thi','inch','ha','width','height']
spl_words1 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','he','al','ae','cv','de','fo','ie','ma','na','op','re','xy','yz','ha','ig','id','id:']
spl_words2 = ['let','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','yes','no','yesno', 'was','the', 'or', 'me','id', 'you', 'my', 'that', 'no', 'your', 'them', 'out', 'do','msg','if', 'ur', 'to', 'a', 'the', 'and', 'is','of', 'we', 'are', 'it', 'am', 'for', 'by', 'they', 'in', 'at', 'this', 'do', 'on', 'have', 'but', 'all', 'be', 'any', 'so']
spl_punct = ['(',')',':','=','??','..','\\','//','!!','@@','##','$$','%%','&&','((','))',':)',':(',':|',':/','./','[]','}{','{}','--','++',':,',').','),','.?',',?','?.','?,','(?)','<>','><','\\.','//.','++','==','_','__','+-','**','-.','+.',')?','??.','??,',':-',':.','.:','()','//,','>:','|?','"?',',-',',,','\\','(,','-?','-:','??:','+:',',-)']
rmv_punctuations = '''!()-[]{}:='"\<>/@#$%^&*_~'''


def clean_text(x):
    if pd.isnull(x) is not True:
        x = re.sub(r'[^\x00-\x7F]','', x)         # Remove all Non ASCII characters
        x = re.sub(r'[\d]','',x)                  # Remove all numbers
        x = re.sub(r'[\n]','',x)                  # Remove New Line Char
        x = re.sub('[!@#$]', '', x)               # Remove any spl Char
        x = re.sub(r'(<)?(\w+@\w+(?:\.\w+)+)(?(1)>)','',x) # Remove Email ID
        x = re.sub(r'^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$','',x)#Remove Email
        x = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) #Remove HTML link
        x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)        #Remove HTML link
        x = re.sub(r'<[^>]*>', '', x)                                 # Remove all HTML tags
        x = re.sub(r'&nbsp;','',x)                                    # Remove &nbsp tags
        x = " ".join(x.split())                   # Remove all white spaces
        x = x.lower()                             # Lower Case
        x = ' '.join([word for word in x.split() if word not in web_words])
        x = ' '.join([word for word in x.split() if word not in spl_words1])
        x = ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)) # Remove repetition of letters
        if len(x) < 1:    # added by Rajesh Rajamani to handle nulls
            x = "Blank"
            return x
        else:
            return x


def strip_text(x):                   # To clean special punctuation marks
    if pd.isnull(x) is not True:
        x = x.strip()
        if len(x) < 1:    # added by Rajesh Rajamani to handle nulls
            x = "Blank"
            return x
        else:
            return x



def clean_text4(x):                   # To clean special punctuation marks
    if pd.isnull(x) is not True:
        x = ' '.join([word for word in x.split() if word not in spl_punct])
        if len(x) < 1:    # added by Rajesh Rajamani to handle nulls
            x = "Blank"
            return x
        else:
            return x


def clean_text5(x):
    if pd.isnull(x) is not True:
        x = ' '.join([word for word in x.split() if word not in cachedStopWords]) # Remove Stop Words
        x = ' '.join([word for word in x.split() if word not in spl_words2])      # Remove Special words
    if len(x) < 1:    # added by Rajesh Rajamani to handle nulls
        x = "Blank"        
        return x
    else:
        return x
        


def rmv_punct(line):                    # Remove Punctuation other than ,.?
    cwrd = ""
    for wrd in line:
        if wrd not in rmv_punctuations:
            cwrd = cwrd + wrd
    return cwrd


def clean_text_1(line):                 # http, //, www etc
    cwrd = []
    for wrd in line:
        if wrd not in web_words:
            cwrd.append(wrd)
    return cwrd

def clean_text_2(line):                 # a,b,c,d etc
    cwrd = []
    for wrd in line:
        if wrd not in spl_words1:
            cwrd.append(wrd)
    return cwrd

def clean_Stp_Wrd(line):                # Stop Words
    cwrd = []
    for wrd in line:
        if wrd not in cachedStopWords:
            cwrd.append(wrd)
    return cwrd


def clean_punct(line):                  # .,;,-,+, etc
    cwrd = []
    for wrd in line:
        if wrd not in list(punctuation):
            cwrd.append(wrd)
    return cwrd


def clean_spl_punct(line):              # ///,+++,---, etc
    cwrd = []
    for wrd in line:
        if wrd not in spl_punct:
            cwrd.append(wrd)
    return cwrd


'''-----------------------------------------------------------------------------------------'''
'''---------  1. Replacing RE & Correcting Words (I've -> I have)  ---------------------------------'''
'''-----------------------------------------------------------------------------------------'''

replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'can not'),
(r'i\'m', 'i am'),
(r'&amp;', 'and'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')
]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        if pd.isnull(s) is not True:
           for (pattern, repl) in self.patterns:
               (s, count) = re.subn(pattern, repl, s)
        else:
           s = "Blank"
        return s

replacer1 = RegexpReplacer()

'''-----------------------------------------------------------------------------------------'''
'''---------- 2.  Repeating Char (Hellooooo -> Helloo)   -----------------------------------'''
'''-----------------------------------------------------------------------------------------'''
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w)\3(\w*)')
        self.repl = r'\1\2\3\4'
    def replace(self, word):
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

replacer2 = RepeatReplacer()

'''-----------------------------------------------------------------------------------------'''
'''----------- 3. Replace Short words (bday -> BirthDay) -----------------------------------'''
'''-----------------------------------------------------------------------------------------'''
class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)

    def replace_sent(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            word = self.replace(word)
            words.append(word)
            i += 1
        return words

class CsvWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)


replacer3 = CsvWordReplacer(smartstoppath+"\\Data\\includes\\sms_word.csv")

'''----------------------------------------------------------------------------------'''
'''--------  Replacing negations with antonyms   -----------------------------------------'''
'''----------------------------------------------------------------------------------'''

class AntonymReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)
    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words


class CsvWordReplacer1(AntonymReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            word_map[word] = syn
        super(CsvWordReplacer1, self).__init__(word_map)


replacer_ar = CsvWordReplacer1(smartstoppath+"\\Data\\includes\\antonyms.csv")



'''-----------------------------------------------------------------------------------------'''
'''----------     Word Lemmatizer  (Ex Believes --> Belief , Absolutely --> Absolute   -----'''
'''-----------------------------------------------------------------------------------------'''
def mywnl(se):
    wnl= WordNetLemmatizer()
    return [wnl.lemmatize(k) for k in se]

'''-----------------------------------------------------------------------------------------'''
'''----------------------  TOKENIZE    ---------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
tockenizer=WordPunctTokenizer()

sentTokenizer = PunktSentenceTokenizer()

'''-----------------------------------------------------------------------------------------'''
'''----------------- Extracting Positive & Negative Words   ----------------------------------'''
'''-----------------------------------------------------------------------------------------'''
"""
Pos_words=open("Data\\includes\\positive-words.txt","r")
Neg_words=open("Data\\includes\\negative-words.txt","r")
Emp_words=open("Data\\includes\\emphasis-words.txt","r")
Not_words=open("Data\\includes\\negation-words.txt","r")
Neg_high_words=open("Data\\includes\\negative-high-words.txt","r")
Pos_Emotion = open("Data\\includes\\positive_emotion.txt","r")
Neg_Emotion = open("Data\\includes\\negative_emotion.txt","r")


Pos_words=Pos_words.read().split(",")
Neg_words=Neg_words.read().split(",")
Neg_high_words=Neg_high_words.read().split(",")
Emp_words=Emp_words.read().split(",")
Not_words=Not_words.read().split(",")
Pos_Emotion= Pos_Emotion.read().split(",")
Neg_Emotion= Neg_Emotion.read().split(",")

"""

#assign cursor for loading various dictionaries
cursor = database_connect() # returns cursor

#load dictionaries
Pos_words = get_dict(cursor,"select tokenword from Dictionary where positivewordflag=1 ") # to be replaced with procedure 
Neg_words = get_dict(cursor,"select tokenword from Dictionary where negativewordflag=1 ") # to be replaced with procedure 
Emp_words = get_dict(cursor,"select tokenword from Dictionary where Emphasiswordflag=1 ") # to be replaced with procedure 
Not_words = get_dict(cursor,"select tokenword from Dictionary where negationwordflag=1 ") # to be replaced with procedure 
Neg_high_words =get_dict(cursor,"select tokenword from Dictionary where negativehighwordflag=1 ") # to be replaced with procedure 
Pos_Emotion = get_dict(cursor,"select tokenword from Dictionary where negativeemotionflag=1 ") # to be replaced with procedure 
Neg_Emotion = get_dict(cursor,"select tokenword from Dictionary where positiveemotionflag=1 ") # to be replaced with procedure 

# get the dictionaries loaded as list ( as this function needs an ordering in the list for iteration)    
PosWords = get_list(cursor,"select tokenword from Dictionary where positivewordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
NegWords = get_list(cursor,"select tokenword from Dictionary where negativewordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list  
EmpWords = get_list(cursor,"select tokenword from Dictionary where Emphasiswordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
NotWords = get_list(cursor,"select tokenword from Dictionary where negationwordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
NegHighWords =get_list(cursor,"select tokenword from Dictionary where negativehighwordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 


#close cursor
cursor.close()

"""Word Bags 
pWrd = []   # Positive Word List
nWrd = []   # Negative Word List
peWrd = []  # Positive Emotion
neWrd = []  # Negative Emotion

def posWrd(text):
    pWrd = []
    for token in text:
        if token in Pos_words:
            pWrd.append(str(token))
    return pWrd

def negWrd(text):
    nWrd = []
    for token in text:
        token=str(token)
        if token in Neg_words:
            nWrd.append(str(token))
    return nWrd


def posEmoWrd(text):
    peWrd = []
    for token in text:
        if token in Pos_Emotion:
            peWrd.append(str(token))
    return peWrd

def negEmoWrd(text):
    neWrd = []
    for token in text:
        token=str(token)
        if token in Neg_Emotion:
            neWrd.append(str(token))
    return neWrd

"""

'''-----------------------------------------------------------------------------------------'''
'''----------------- Sentiment Score --- Positive & Negative Score   -----------------------'''
'''-----------------------------------------------------------------------------------------'''

scr1 = 0
def SentiScoreNew(text):
    scr1 = 0
    N = len(text)
    for i in range(N):
        if text[i] in PosWords:
            scr1 = scr1 + 1
            if i >1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 + 1
                if text[i-1] in NotWords:
                    scr1 = scr1 - 2
        elif text[i] in NegWords:
            scr1 = scr1 - 1
            if text[i] in NegHighWords:
                scr1=scr1-2
            if i > 1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 - 1
                if text[i-1] in NotWords:
                    scr1 = scr1 + 2
    return scr1

scr2 = 0
def PosSentiScore(text):
    scr2 = 0
    N = len(text)
    for i in range(N):
        if text[i] in PosWords:
            scr2 = scr2 + 1
            if i >1:
                if text[i-1] in EmpWords:
                    scr2 = scr2 + 1
                if text[i-1] in NotWords:
                    scr2 = scr2 - 2
    return scr2

scr3 = 0
def NegSentiScore(text):
    scr3 = 0
    N = len(text)
    for i in range(N):
        if text[i] in NegWords:
            scr3 = scr3 - 1
            if i > 1:
                if text[i-1] in EmpWords:
                    scr3 = scr3 - 1
                if text[i-1] in NotWords:
                    scr3 = scr3 + 2
    return scr3

'''--------------------------------------------'''
'''--------    Sentiment           --------------'''
'''--------------------------------------------'''
def SentimentClassifier(data_frame):
    #combines the functionality of both SentiMent and SentiConfidence
    #Accepts a data frame

    #SentiMent

    data_frame['Senti']='Neutral'
    data_frame.loc[data_frame['Score']< 0,'Senti'] ='Negative'
    data_frame.loc[data_frame['Score']> 0,'Senti'] ='Positive'

    #SentiConfidence

    data_frame['ConfidenceLevel']='0'
    data_frame.loc[data_frame['Score']< -20,'ConfidenceLevel'] ='99'
    data_frame.loc[data_frame['Score']> -20,'ConfidenceLevel'] ='90'
    data_frame.loc[data_frame['Score']> -10,'ConfidenceLevel'] ='70'
    data_frame.loc[data_frame['Score']> -5,'ConfidenceLevel'] ='40'
    data_frame.loc[data_frame['Score']> -1,'ConfidenceLevel'] ='20'
    data_frame.loc[data_frame['Score']> 0,'ConfidenceLevel'] ='10'
    data_frame.loc[data_frame['Score']> 2,'ConfidenceLevel'] ='30'
    data_frame.loc[data_frame['Score']> 5,'ConfidenceLevel'] ='70'
    data_frame.loc[data_frame['Score']> 10,'ConfidenceLevel'] ='90'
    data_frame.loc[data_frame['Score']> 20,'ConfidenceLevel'] ='99'

def SentiMent(score):
    if score > 0:
        return 'Positive'
    elif score <0:
        return 'Negative'
    else:
        return 'Neutral'
   

'''-----------------------------------------------------------------------------------------'''
'''------------------- POS Tagging   --------------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
from nltk.corpus import treebank
from nltk.tag import DefaultTagger, UnigramTagger

train_sents = treebank.tagged_sents()[:3000]

tagger1 = DefaultTagger('NN')
tagger2 = UnigramTagger(train_sents, backoff=tagger1)
'''-----------------------------------------------------------------------------------------'''
'''------------------- Chunking with POS Tagging ---------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
chunker = RegexpParser(r'''
    NP:
        {<DT>?<NN.*><VB.*><DT.*>?<NN.*>}
        {<DT>?<NN.*><IN><DT><NN.*>}
        {<NN.*><VB.*><NN.*>}
    ''')

chunker2 = RegexpParser(r'''
    Phrase:
        {<JJ.*><NN.*>}
        {<RB><JJ>^<NN.*>}
        {<JJ><JJ>^<NN.*>}
        {<NN.*><JJ>^<NN.*>}
        {<RB.*><VB.*>}
    ''')


chunkerPOS = RegexpParser(r'''
    Noun:
        {<DT.*><NN.*><IN.*><DT><NN.*>}
        {<NN.*><NN.*>?<NN.*>?<NN.*>}
        {<NN.*><IN.*><NN.*>}
    Verb:
        {<DT.*><VB.*><NN.*>}
        {<VB.*><NN.*>}
    Adj:
        {<JJ.*>}
    ''')



'''-------------------------------------------------------------------------------------'''
'''-------------------  Extracting Root Cause   -------------------------------------------'''
'''-------------------------------------------------------------------------------------'''
rc = []
def FeatureExtractor(tree):
    rc = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            rc.append(str(untag(subtree)))
    return rc

rc1 = []
def NounExtractor(tree):
    rc1 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Noun':
            rc1.append(str(untag(subtree)))
    return rc1

rc2 = []
def VerbExtractor(tree):
    rc2 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Verb':
            rc2.append(str(untag(subtree)))
    return rc2

rc3 = []
def AdjExtractor(tree):
    rc3 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Adj':
            rc3.append(str(untag(subtree)))
    return rc3

rc4 = []
def PhraseExtractor(tree):
    rc4 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Phrase':
            rc4.append(str(untag(subtree)))
    return rc4










'''-----------------------------------------------------------------------------------------'''
'''----------------------Customized POS Tagging -----------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
ptagger = DefaultTagger('PWD')
ntagger = DefaultTagger('NWD')
etagger = DefaultTagger('EMP')

tag_pos = ptagger.tag(PosWords) # changed to list version of Dictionary
tag_neg = ntagger.tag(NegWords) # changed to list version of Dictionary
tag_emp = etagger.tag(EmpWords) # changed to list version of Dictionary

tag_wrd = tag_pos + tag_neg + tag_emp
tag_wrd_dict = dict(tag_wrd)


tagger5 = UnigramTagger(model = tag_wrd_dict, backoff= tagger2)


'''-----------------------------------------------------------------------------------------'''
'''------------------- Chunking with POS Tagging ---------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
chunker1 = RegexpParser(r'''
    PWD:
        {<PWD><NN.*>}
        {<PWD><JJ.*>}
        {<PWD><VB.*>}
        {<RB.*><PWD>}
        {<NN.*><PWD>}
    NWD:
        {<NWD><NN.*>}
        {<NWD><JJ.*>}
        {<NWD><VB.*>}
        {<RB.*><NWD>}
        {<NN.*><NWD>}
    ''')

'''--------------  Extracting Positive & Negative Sentence  ------------------------------------------'''

ps = []
def PosExtractor(tree):
    ps = []
    for subtree in tree.subtrees():
        if subtree.label() == 'PWD':
            ps.append(str(untag(subtree)))
    return ps

ns = []
def NegExtractor(tree):
    ns = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NWD':
            ns.append(str(untag(subtree)))
    return ns

'''-----------------------------------------------------------------------------------------'''
'''------------------- Create Corpus & Frequency Matrix ------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''

Wrd = []
def WordCorpus(text):
    for token in text:
        Wrd.append(str(token))
    return Wrd

pWrd = []
def pWordCorpus(text):
    for token in text:
        #token=str(token)
        if token in Pos_words:
            pWrd.append(str(token))
    return pWrd

nWrd = []
def nWordCorpus(text):
    for token in text:
        token=str(token)
        if token in Neg_words:
            nWrd.append(str(token))
    return nWrd


'''-----------------------------------------------------------------------------------------'''
'''------------------- Categories ----------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''

cat = pd.read_csv(smartstoppath+'\\Data\\categories\\categories.csv',encoding='latin-1')
catg = pd.read_csv(smartstoppath+'\\Data\\categories\\categorylist.csv',encoding='latin-1')
#Categories 1
cat1A = cat['cat1A'].tolist()
cat1B = cat['cat1B'].tolist()
cat1C = cat['cat1C'].tolist()

#Categories 2
cat2A = cat['cat2A'].tolist()
cat2B = cat['cat2B'].tolist()
cat2C = cat['cat2C'].tolist()

#Categories 3
cat3A = cat['cat3A'].tolist()
cat3B = cat['cat3B'].tolist()
cat3C = cat['cat3C'].tolist()

#Categories 4
cat4A = cat['cat4A'].tolist()
cat4B = cat['cat4B'].tolist()
cat4C = cat['cat4C'].tolist()

#Categories 5
cat5A = cat['cat5A'].tolist()
cat5B = cat['cat5B'].tolist()
cat5C = cat['cat5C'].tolist()




def CatScore1(text):
    scr = 0
    for token in text:
        if token in cat1A:
            scr=scr+1
        if token in cat1B:
            scr=scr+0.5
        if token in cat1C:
            scr=scr-1
    return (scr)

def CatScore2(text):
    scr = 0
    for token in text:
        if token in cat2A:
            scr=scr+1
        if token in cat2B:
            scr=scr+0.5
        if token in cat2C:
            scr=scr-1
    return (scr)

def CatScore3(text):
    scr = 0
    for token in text:
        if token in cat3A:
            scr=scr+1
        if token in cat3B:
            scr=scr+0.5
        if token in cat3C:
            scr=scr-1
    return (scr)
    


def CatScore4(text):
    scr = 0
    for token in text:
        if token in cat4A:
            scr=scr+1
        if token in cat4B:
            scr=scr+0.5
        if token in cat4C:
            scr=scr-1
    return (scr)

def CatScore5(text):
    scr = 0
    for token in text:
        if token in cat5A:
            scr=scr+1
        if token in cat5B:
            scr=scr+0.5
        if token in cat5C:
            scr=scr-1
    return (scr)
	
'''-----------------------------------------------------------------------------------------'''
'''--------------------------------- ST Tagger -----------------------------------'''
'''-----------------------------------------------------------------------------------------'''	
def ORGANIZATION_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "ORGANIZATION":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd
   
def PERSON_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "PERSON":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd
    
def Time_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "TIME":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd

def Location_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "LOCATION":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd


def Money_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "MONEY":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd

def Percent_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "PERCENT":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd

def Date_tag(line):                    
    cwrd = "Tag: "
    for tag, chunk in groupby(line, lambda x:x[1]):
        if tag == "DATE":
            if cwrd == "Tag: ":
                cwrd=" ".join(w for w, t in chunk)
            else:
                cwrd=cwrd, " ".join(w for w, t in chunk)
    return cwrd
    

def tagging(x):                   # To clean special punctuation marks
    if pd.isnull(x) is not True:
        #x = ' '.join([word for word in x.split() if word not in spl_punct])
        x1 = st7.tag(x.split())
    return x1	
	
	

'''-----------------------------------------------------------------------------------------'''
'''------------------- Main Function : Run All Functions -------------------------------'''
'''-----------------------------------------------------------------------------------------'''


if __name__ == "__main__":
   print (str((sys.argv[1:4]) ))
   #print(sys.argv[0])
var_batchid ='Health'
   

   
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#print (var_batchid)

try:
	LogFileHandler(var_batchid)
	#keeda

	start_time = time.clock()
	#os.chdir('D:\\Pankaj\\Projects\\WebSite_TabBrowsing\\Python')

	# test = pd.read_csv('Data\\Barclays1.csv',encoding='latin-1')                    # Reading CSV data
	#for testing "20160225113129"
	batchid = var_batchid # should be passed by web application for Python engine to process 
	#sql_query = "select DateCol as date,TextCol as text,RecordID as RecordID from TextDataImport where RecordID < 1000 and BatchID = '%s'" % batchid
	#test = GetRecordsForAnalysisFromSQL(batchid)

	#prepare the sql stringl
	PrepSQLForRecords(batchid)
	#print("949")
  
  
	sql_string = "exec USP_GetRecordsForAnalysis '%s'" % batchid    
  
	test=pd.read_csv('C:\\Users\\ii00061002\\Desktop\\Testfile.csv',encoding='latin-1')
	#print("955")

	#rename columns to suit references further 
	#DateColumn >>> date , TextColumn >>> text

	test.rename(columns={'DateColumn': 'date', 'TextColumn': 'text'}, inplace=True)
 
	elapsedtime = print_elapsed_time(start_time,"Import")

	'''---------------------------------------------------------------------------------'''
	test['RealText'] = test['text']                             # Real Text
	#test['text'] = test['text'].apply(clean_text)               # Cleaning Text

	elapsedtime = print_elapsed_time(elapsedtime,"Cleaning Text")

	#test['text'] = test['text'].apply(replacer1.replace)        # I've -> I have

	elapsedtime = print_elapsed_time(elapsedtime,"Replacer 1")

	#test['text'] = test['text'].apply(replacer2.replace)        # Helloooo -> Helloo

	elapsedtime = print_elapsed_time(elapsedtime,"Replacer 2")
	#test['text'] = test['text'].apply(clean_text4)              # Remove Special Punctuation

	elapsedtime = print_elapsed_time(elapsedtime,"Special Punctuations1")
	#test['text'] = test['text'].apply(rmv_punct)                # Remove Special Punctuation

	elapsedtime = print_elapsed_time(elapsedtime,"Special Punctuations2")

	'''---------------------------------------------------------------------------------'''
	''' Text is cleaned data but with Punctuation and preposition and stop words. Text1 is completely cleaned'''
	test['text_AllCleaned'] = test['text'].apply(clean_text5)               # Final Cleaning Text
	elapsedtime = print_elapsed_time(elapsedtime,"Final CLeaning")

	test['Token1'] =test['text_AllCleaned'].apply(tockenizer.tokenize)      # Tokenizer
	elapsedtime = print_elapsed_time(elapsedtime,"Tokenizer")

	test['Token1'] =test['Token1'].apply(clean_text_1)            # Clean text level 2
	elapsedtime = print_elapsed_time(elapsedtime,"CleanText Level 2")

	#test['Token1'] =test['Token1'].apply(clean_text_2)            # Clean Few Stop Words
	elapsedtime = print_elapsed_time(elapsedtime,"CleanText StopWords")

	test['Token1'] =test['Token1'].apply(clean_Stp_Wrd)           # Cleaned all English Stop Words
	elapsedtime = print_elapsed_time(elapsedtime,"All English Stopwords")

	test['Token1'] =test['Token1'].apply(clean_punct)             # Cleaned Punctuation
	elapsedtime = print_elapsed_time(elapsedtime,"CleanPunctuation")

	test['Token1'] =test['Token1'].apply(clean_spl_punct)         # Cleaned Special Punctuation
	elapsedtime = print_elapsed_time(elapsedtime,"SPL_Punctuation")

	'''---------------------------------------------------------------------------------'''
	test['Token'] =test['text'].apply(tockenizer.tokenize)      # Tokenizer
	elapsedtime = print_elapsed_time(elapsedtime,"Another tokenizer")

	test['Token'] =test['Token'].apply(clean_spl_punct)         # Cleaned Special Punctuation
	elapsedtime = print_elapsed_time(elapsedtime,"SPL_Punctuation_Again")

	test['Token'] =test['Token'].apply(replacer_ar.replace_negations)       # Replace Negation with Antonyms
	elapsedtime = print_elapsed_time(elapsedtime,"Negations")

	test['Token'] =test['Token'].apply(replacer3.replace_sent)  # bday -> BirthDay
	elapsedtime = print_elapsed_time(elapsedtime,"Replace_Abbrievations")


	### Temp comment
	#test['Token'] =test['Token'].apply(mywnl)                   # WordNetLemmatizer ( Believes -> Belief)
	elapsedtime = print_elapsed_time(elapsedtime,"Word Net Lemmatizer")
	'''---------------------------------------------------------------------------------'''  		
	
	
 
      ###Ajay### test['Tag'] =test['Token'].apply(tagger2.tag)               # NER Tagging
	###Ajay### test['Tag'] = test['RealText'].apply(tagging)  
	test['Tag'] = 0   
	#elapsedtime = print_elapsed_time(elapsedtime,"NER Tagging")

	###Ajay### test['Tree'] =test['Tag'].apply(chunker.parse)              # Organization
	###Ajay### 	test['Tree'] =test['Tag'].apply(ORGANIZATION_tag) 
 
	#elapsedtime = print_elapsed_time(elapsedtime,"Organization")

	###Ajay### test['Tree2'] =test['Tag'].apply(chunker2.parse)            # Person
	###Ajay### 	test['Tree2'] =test['Tag'].apply(PERSON_tag)  
	#elapsedtime = print_elapsed_time(elapsedtime,"Person")

	###Ajay### test['TreePOS'] =test['Tag'].apply(chunkerPOS.parse)        # Time
	###Ajay### 	test['TreePOS'] =test['Tag'].apply(Time_tag)
	#elapsedtime = print_elapsed_time(elapsedtime,"Time")

	'''---------------------------------------------------------------------------------'''
	###Ajay### test['Tag1'] =test['Token'].apply(tagger5.tag)              # Location
	###Ajay### 	test['Tag1'] =test['Tag'].apply(Location_tag)
	#elapsedtime = print_elapsed_time(elapsedtime,"Location")

	###Ajay### test['Tree1'] =test['Tag1'].apply(chunker1.parse)           # Money
	###Ajay### 	test['Tree1'] =test['Tag'].apply(Money_tag)
	#elapsedtime = print_elapsed_time(elapsedtime,"Money")

	###Ajay### test['PosCause'] =test['Tree1'].apply(PosExtractor)         # Percent
	###Ajay### 	test['PosCause'] =test['Tag'].apply(Percent_tag)
	#elapsedtime = print_elapsed_time(elapsedtime,"Percent")

	###Ajay### test['NegCause'] =test['Tree1'].apply(NegExtractor)         # Date
	###Ajay### 	test['NegCause'] = test['Tag'].apply(Date_tag)
	#elapsedtime = print_elapsed_time(elapsedtime,"Date")

      ###Ajay Temp###
	test['Tree'] = 0
	test['Tree2'] = 0
	test['TreePOS'] = 0
	test['Tag1'] = 0     
	test['Tree1'] = 0
	test['PosCause'] = 0  
	test['NegCause'] = 0  
 
# ----------------------------------------------------------------------------------------------------------------------

#### Code for personality prediction #### Ajay ####

	x_test1 = np.array(test['text_AllCleaned'])
 
	df = pd.read_csv(smartstoppath+'\\Data\\categories\\Personality1_Train.csv',encoding='latin1')
 
	X_train = df.recipe_name

	y_train_text = df.classes.str.split('|')

	target_names = ['Inventive/Curious','Consistent/Cautious','Easy-going/Careless','Efficient/Organized','Solitary/Reserved','Outgoing/Energetic','Analytical/Detached','Friendly/Compassionate','Sensitive/Nervous','Secure/Confident']

	lb = preprocessing.MultiLabelBinarizer()
	Y = lb.fit_transform(y_train_text)
 
	classifier = Pipeline([
                            ('vectorizer', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', OneVsRestClassifier(LinearSVC()))])
	#load model
	classifier.fit(X_train, Y)
	#load model
	predicted = classifier.predict(x_test1)
	all_labels = lb.inverse_transform(predicted)
	prediction = pd.DataFrame(all_labels,columns=['A','B','C','D','E'])
	
	prediction['period'] = prediction[prediction.columns[0:]].apply(lambda x: '~'.join(x), axis=1) 
 
     #vector transform
	test['RootCause']=prediction['period']
	elapsedtime = print_elapsed_time(elapsedtime,"Personality Detection")

####  Ajay  ####


	###Ajay### test['Phrase'] =test['Tree2'].apply(PhraseExtractor)        # Two Word Phrase Extractor
	test['Phrase']=0
	elapsedtime = print_elapsed_time(elapsedtime,"Phrase Extractor")

	###Ajay### test['Noun'] = test['TreePOS'].apply(NounExtractor)         # Noun Extractor
	test['Noun'] =0
	elapsedtime = print_elapsed_time(elapsedtime,"Noun Extractor")

	###Ajay### test['Verb'] = test['TreePOS'].apply(VerbExtractor)         # Verb Extractor
	test['Verb'] =0
	elapsedtime = print_elapsed_time(elapsedtime,"Verb Extractor")

	###Ajay### test['Adj'] = test['TreePOS'].apply(AdjExtractor)           # Adjective Extractor
	test['Adj'] =0
	elapsedtime = print_elapsed_time(elapsedtime,"Adjective Extractor")
      
       
	
	
	
	'''---------------------------------------------------------------------------------'''
	###ajay### test['cat1'] =test['Token'].apply(CatScore1)             # 1st Category
	###ajay### test['cat2'] =test['Token'].apply(CatScore2)             # 2nd Category
	###ajay### test['cat3'] =test['Token'].apply(CatScore3)             # 3rd Category
	###ajay### test['cat4'] =test['Token'].apply(CatScore4)             # 4th Category
	###ajay### test['cat5'] =test['Token'].apply(CatScore5)             # 5th Category

	###ajay### elapsedtime = print_elapsed_time(elapsedtime,"Cat Scores")
	'''-------------------------------------------------------------------------------'''
	test['list'] = 0 #test[['cat1', 'cat2', 'cat3','cat4', 'cat5']].idxmax(axis=1)
      
	'''---------------------------------------------------------------------------------'''

	""" OldMethods
	test['PositiveWords'] =test['Token'].apply(posWrd)          # Positive Word Extractor
	test['NegativeWords'] =test['Token'].apply(negWrd)          # Negative Word Extractor

	test['PositiveEmotion'] =test['Token'].apply(posEmoWrd)     # Positive Word Extractor
	test['NegativeEmotion'] =test['Token'].apply(negEmoWrd)     # Negative Word Extractor
	"""
	#Call Database_to_dictionary function 
	cursor = database_connect() # returns cursor

	#load as dictionary for workdbuckets
	Pos_words = get_dict(cursor,"select tokenword from Dictionary where positivewordflag=1 ") # to be replaced with procedure 
	Neg_words = get_dict(cursor,"select tokenword from Dictionary where negativewordflag=1 ") # to be replaced with procedure 
	Emp_words = get_dict(cursor,"select tokenword from Dictionary where Emphasiswordflag=1 ") # to be replaced with procedure 
	Not_words = get_dict(cursor,"select tokenword from Dictionary where negationwordflag=1 ") # to be replaced with procedure 
	Neg_high_words =get_dict(cursor,"select tokenword from Dictionary where negativehighwordflag=1 ") # to be replaced with procedure 
	Pos_Emotion = get_dict(cursor,"select tokenword from Dictionary where positiveemotionflag=1 ") # to be replaced with procedure 
	Neg_Emotion = get_dict(cursor,"select tokenword from Dictionary where negativeemotionflag=1 ") # to be replaced with procedure 


	# get the dictionaries loaded as list for sentiment score iteration ( as this function needs an ordering in the list for iteration)    
	PosWords = get_list(cursor,"select tokenword from Dictionary where positivewordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
	NegWords = get_list(cursor,"select tokenword from Dictionary where negativewordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list  
	EmpWords = get_list(cursor,"select tokenword from Dictionary where Emphasiswordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
	NotWords = get_list(cursor,"select tokenword from Dictionary where negationwordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
	NegHighWords =get_list(cursor,"select tokenword from Dictionary where negativehighwordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 



	cursor.close()

	##Word Bucket from tokens
	#de-activated this line as this already available 
	#test['Token']=test['text'].apply(tockenizer.tokenize)

	elapsedtime = print_elapsed_time(elapsedtime,"Cat Scores")

	word_bucket_classifier = partial(WordBucket, dictionary=Pos_words)  # Positive_Words
	test['PositiveWords'] = test['Token'].apply(word_bucket_classifier)

	word_bucket_classifier = partial(WordBucket, dictionary=Neg_words)  # Negative words
	test['NegativeWords'] = test['Token'].apply(word_bucket_classifier)

	###ajay### word_bucket_classifier = partial(WordBucket, dictionary=Emp_words)  # Emphasis Words
	###ajay### test['EmphasisWords'] = test['Token'].apply(word_bucket_classifier)

	###ajay### word_bucket_classifier = partial(WordBucket, dictionary=Not_words)  # Negation Words
	###ajay### test['NegationWords'] = test['Token'].apply(word_bucket_classifier)

	###ajay### word_bucket_classifier = partial(WordBucket, dictionary=Neg_high_words)  # Negative High Words
	###ajay### test['NegativeHighWords'] = test['Token'].apply(word_bucket_classifier)

	word_bucket_classifier = partial(WordBucket, dictionary=Pos_Emotion)  # Positive Emotion Words
	test['PositiveEmotion'] = test['Token'].apply(word_bucket_classifier)

	word_bucket_classifier = partial(WordBucket, dictionary=Neg_Emotion)  # Negative Emotion Words
	test['NegativeEmotion'] = test['Token'].apply(word_bucket_classifier)

	elapsedtime = print_elapsed_time(elapsedtime,"Word Bucket Classification")



	test['PosScore'] = test['Token'].apply(PosSentiScore)
	test['NegScore'] = test['Token'].apply(NegSentiScore)


	elapsedtime = print_elapsed_time(elapsedtime,"Sentiment Scores +-")

	test['Score'] =test['Token'].apply(SentiScoreNew)           # New Logic for Sentiment Score
	elapsedtime = print_elapsed_time(elapsedtime,"Sentiment_Score")

	#Optimised Sentiment Classification
	SentimentClassifier(test)
	elapsedtime = print_elapsed_time(elapsedtime,"Optimized Sentiments")

	#Deactivated as this is carried oout in the Optimized Sentiment Classification Step
	"""
	test['Senti'] =test['Score'].apply(SentiMent)               # Sentiment
	test['ConfidenceLevel'] =test['Score'].apply(SentiConfidence)# Confidence Level of Sentiment
	"""

	test['BagOfWrd'] = 0 ###Ajay### test['Token'].apply(bag_of_words)        # Bag of Words
     
	###Ajay### elapsedtime = print_elapsed_time(elapsedtime,"Bag Of Words")

	test['SentiClass'] = 0 ###Ajay### test['BagOfWrd'].apply(nb_classifier.classify)        # NB Classify
     
	###Ajay### elapsedtime = print_elapsed_time(elapsedtime,"NB Classify")
	



	test['one'] = 1                                             # assigning value one to each rows
	
	#Rake 

	test['sentencelist']=test['text_AllCleaned'].apply(getsplitsentences)
	test['phraselist']=test['sentencelist'].apply(getphraselist)
	test['keywordcandidates']=test['phraselist'].apply(getcandidatescores)
	elapsedtime = print_elapsed_time(elapsedtime,"RAKE Process")
 
    
	
	#predicting sentiment with SVM
	test['TokenText'] = test['Token1'].apply(lambda x: listtostr(x))
	# the following steps should be referring to TokenText earlier there were Token1 
	#changed by Rajesh Rajamani on 27-May-2016
	test['TokenTextClean'] = test['TokenText'].apply(lambda x: re.sub('[^a-zA-Z,]',' ',x))
	test['TokenTextClean'] = test['TokenText'].apply(lambda x: stemming(x))
	elapsedtime = print_elapsed_time(elapsedtime,"Sentiment Prediction with SVM Preparation")
	
  
	#########################################################################
	#######################  Emotion & Sentiment model ######################


	#load model for sentiment
	#clf2=joblib.load(smartstoppath+"\\Model\\Without_Vectoriser_Sentiment_model.pkl")
	#elapsedtime = print_elapsed_time(elapsedtime,"Model Load for SVM")

     #load model for Emotion
	#clf2_e=joblib.load(smartstoppath+"\\Model\\Without_Vectoriser_Emotion_model.pkl")
	#elapsedtime = print_elapsed_time(elapsedtime,"Model Load for SVM Emotion")	
	
	#vector transform
	#x_test = np.array(test['TokenTextClean'])
	
	#test['PredictedSentiment']=clf2.predict(x_test)
	#elapsedtime = print_elapsed_time(elapsedtime,"Completed Sentiment Prediction with Model")
    
	#test['PredictedSentiment_confidense']=clf2_e.predict(x_test)
	#elapsedtime = print_elapsed_time(elapsedtime,"Completed Emotion Prediction with Model")
  

	#load model for sentiment
	clf2=joblib.load(smartstoppath+"\\Model\\Without_Vectoriser_Sentiment_Emotion.pkl")
	elapsedtime = print_elapsed_time(elapsedtime,"Model Load for SVM")

	x_test = np.array(test['TokenTextClean'])

	test['Predicted']=clf2.predict(x_test)
	test['PredictedSentiment']=test['Predicted'].str.split('~').str.get(0)
	test['PredictedSentiment_confidense']=test['Predicted'].str.split('~').str.get(1)

	test['PredictedSentiment']=test['PredictedSentiment'].apply(strip_text)  
	test['PredictedSentiment_confidense']=test['PredictedSentiment_confidense'].apply(strip_text)  

	test = test.drop(['Predicted'], axis=1)
	elapsedtime = print_elapsed_time(elapsedtime,"Completed Sentiment & Emotion Prediction with Model")


	'''-----------------------------------------------------------------------------'''
	'''------------------   End of Running Function       --------------------------'''
	'''-----------------------------------------------------------------------------'''

	'''-----------------------------------------------------------------------------------------'''
	'''------------ Printing & Extracting File  ----------------------------------------------------------'''
	'''-----------------------------------------------------------------------------------------'''
	#setting data column to 1 this should be deactivated when data is available
	#test['date']='1'
 
 
      
      
	#renaming the columns back to Original name from the database - Not required
	#test.rename(columns={'date': 'DateColumn', 'text': 'AnalysisColumn'}, inplace=True)
	file_name_with_path = smartstoppath+"\\result\\TextAnalysisFile" + var_batchid+".csv"
	test.to_csv(file_name_with_path, sheet_name='sheet1', index=False, header=True , quoting=0) #quoting=0 to enclose in double quotes


	elapsedtime = print_elapsed_time(elapsedtime,"Base File to CSV Export")

	""" Final Upload to Database """

	"""
	#prepare data frame for database upload NaN values will have to be replaced with a filler
	test.fillna("None",inplace=True)
	#length check
	test.loc[test['text_AllCleaned'].str.len() < 1, 'text_AllCleaned'] = "0"
	#null tests
	test = test.where((pd.notnull(test)), None)

	#convert all columns to string
	test.applymap(str)


	#exported_df = pd.read_csv(file_name_with_path,encoding='utf8')
	exported_df = pd.read_csv(open(file_name_with_path,'rU'), encoding='latin-1', engine='c')
	#print(exported_df.head(n=3))
	exported_df.fillna("None",inplace=True)
	exported_df = exported_df.where((pd.notnull(test)), None)

	#Add Batch ID from initial assignment to the dataframe used for upload --Not Required
	#exported_df['FileUMID']=var_FileUMID
	
	#validation
	exported_df = test
	exported_df.to_string
	exported_df.fillna("None",inplace=True)
	exported_df = exported_df.where((pd.notnull(test)), None)
	
	
	#Columns not imported back are deleted - Discussion with AJay and ANil dated 15-Jun-16
	

	exported_df = exported_df.drop(['RealText','TokenText','TokenTextClean'], axis=1)
	
	#Blank columns added for future used
	#exported_df['PredictedSentiment_confidense'] = 0
	exported_df['Model Category']= 0
	exported_df['Model Category Confidence'] = 0
	exported_df['CategoryConfidence'] = 0
	exported_df['KeyWordList'] = 0
	exported_df['Extra1'] = 0
	exported_df['Extra2']= 0
	exported_df['Extra3']= 0
	#testing the file prepared for import back to database
	file_name_with_path1 = smartstoppath+"\\result\\exported_df" + var_FileUMID+".csv"
	exported_df.to_csv(file_name_with_path1, sheet_name='sheet1', index=False, header=True,quoting=0)
	
	exported_df = pd.read_csv(file_name_with_path1,encoding='latin1')
	

	#Table Name
	tableToWriteTo=var_InsertDataTable #"TextData"

	# pyodbc DB_DSN Should be present as a system dsn
	engine = create_engine('mssql+pyodbc://sa:Insight@123@iPrompto_DSN')
	conn = engine.connect()

	# The orient='records' is the key of this, it allows to align with the format mentioned in the doc to insert in bulks.
	listToWrite = exported_df.to_dict(orient='records')
	print("insersion assigned")

	#metadata = sqlalchemy.schema.MetaData(bind=engine,reflect=True)
	metadata = sqlalchemy.schema.MetaData(bind=conn,reflect=True)
	table = sqlalchemy.Table(tableToWriteTo, metadata, autoload=True)

	# Open the session
	#Session = sessionmaker(bind=engine)
	Session = sessionmaker(bind=conn)
	session = Session()
	print("session commit")
 
 

	# Inser the dataframe into the database in one bulk
	conn.execute(table.insert(), listToWrite)
	print("insertion executed")

	# Commit the changes
	session.commit()

	# Close the session
	session.close()

	UpdateBatchFlag(var_FileUMID, 1)

	elapsedtime = print_elapsed_time(elapsedtime,"Processed data stored to database with flag update")
	elapsed_time = time.clock() - start_time
	os.remove(file_name_with_path) 
	os.remove(file_name_with_path1)  	
	print ("Time elapsed: {} seconds".format(elapsed_time))
	#print ("Proceeding further")
"""
except BaseException as e:
	logger.exception(var_FileUMID +','+'Got exception on main handler' + str(e),exc_info=False)
	UpdateBatchFlag(var_FileUMID, 0)


#Updating ProcessedBatchDetails table on successful completion



"""


'''-----------------------------------------------------------------------------'''
'''----------------            Subsetting Data           --------------------------'''
'''-----------------------------------------------------------------------------'''
test_positive = test[test.Senti=='Positive']
test_negative = test[test.Senti=='Negative']
test_neutral = test[test.Senti=='Neutral']


'''-----------------------------------------------------------------------------'''
'''---------------- Word Corpus & Term Frequency Matrix --------------------------'''
'''-----------------------------------------------------------------------------'''
All_Positive = test['PositiveWords'].tolist()
All_Positive = reduce(lambda x,y: x+y,All_Positive)

All_Negative = test['NegativeWords'].tolist()
All_Negative = reduce(lambda x,y: x+y,All_Negative)

all_pos_neg = All_Positive + All_Negative

All_Phrase = test['Phrase'].tolist()
All_Phrase = reduce(lambda  x,y: x+y,All_Phrase)

Pos_Phrase = test_positive['Phrase'].tolist()
Pos_Phrase = reduce(lambda  x,y: x+y,Pos_Phrase)

Neg_Phrase = test_negative['Phrase'].tolist()
Neg_Phrase = reduce(lambda  x,y: x+y,Neg_Phrase)

Neu_Phrase = test_neutral['Phrase'].tolist()
Neu_Phrase = reduce(lambda  x,y: x+y,Neu_Phrase)

All_Noun = test['Noun'].tolist()
All_Noun = reduce(lambda x,y: x+y,All_Noun)

All_Verb = test['Verb'].tolist()
All_Verb = reduce(lambda x,y: x+y,All_Verb)


'''------------------------------------------------------------------------------'''
WrdCrps = test['Token'].tolist()
WrdCrps = reduce(lambda x,y: x+y,WrdCrps)

WrdCrps_cleaned = test['Token1'].tolist()

WrdCrps_cleaned = reduce(lambda x,y: x+y,WrdCrps_cleaned)

'''------------- Pivot for line chart -------------------------------------------'''

pivot2 = pd.pivot_table(test,index=["date", "Senti"], values=["one"], aggfunc=np.sum)
pivot2.to_csv('result\\LineChart.csv')

#Added by Rajesh Rajamani
#Started
'''------------- Pivot for category wise Positive and Negative -------------------------------------------'''
pivot_rc = pd.pivot_table(test,index=["cat_list"], values=["PosScore","NegScore"], aggfunc=np.sum)
pivot_rc.rename(columns={'PosScore': 'positive', 'NegScore': 'negative'}, inplace=True)
pivot_rc.to_csv('result\\rc.csv')

'''------------- Pivot for category wise Positive and Negative BY MONTH -------------------------------------------'''
#this assumes that the date is in yyyy-mm-dd format
test['MonthYear'] = test['date'].str[:7]
pivot_rcm = pd.pivot_table(test,index=["cat_list","MonthYear"], values=["PosScore","NegScore"], aggfunc=np.sum)
pivot_rcm.rename(columns={'PosScore': 'positive', 'NegScore': 'negative'}, inplace=True)
pivot_rcm.to_csv('result\\rc_monthwise.csv')

'''------------- Pivot for category wise Positive and Negative BY Date -------------------------------------------'''
#this assumes that the date is in yyyy-mm-dd format
#test['MonthYear'] = test['date'].str[:7]
pivot_rcd = pd.pivot_table(test,index=["cat_list","MonthYear","date"], values=["PosScore","NegScore"], aggfunc=np.sum)
pivot_rcd.rename(columns={'PosScore': 'positive', 'NegScore': 'negative'}, inplace=True)
pivot_rcd.to_csv('result\\rc_daywise.csv')

'''------------- Pivot for Volume BY Date -------------------------------------------'''
#this assumes that the date is in yyyy-mm-dd format
#test['MonthYear'] = test['date'].str[:7]
pivot_dv = pd.pivot_table(test,index=["cat_list","MonthYear","date"], values=["one"], aggfunc=np.sum)
pivot_dv.rename(columns={'one': 'volume'}, inplace=True)
pivot_dv.to_csv('result\\volume_daywise.csv')

'''------------- Pivot for Volume BY Month BY Cat -------------------------------------------'''
#this assumes that the date is in yyyy-mm-dd format
#test['MonthYear'] = test['date'].str[:7]
pivot_dv = pd.pivot_table(test,index=["cat_list","MonthYear"], values=["one"], aggfunc=np.sum)
pivot_dv.rename(columns={'one': 'volume'}, inplace=True)
pivot_dv.to_csv('result\\volume_monthwise_catwise.csv')

'''------------- CleanedSentences -------------------------------------------'''
cleanedcomment = pd.DataFrame(test['text_AllCleaned'])
cleanedcomment.to_csv('result\\cleanedcomment.csv',index=False)

#Ended




'''--------------- For Pie Chart ----------------------------------------------------'''
sentiment = test['Senti'].tolist()
s1 = []
s1 = Counter(sentiment).most_common(5)

#original: with open('result\\SentimentPi.csv', 'wb') as test_file:
with open('result\\SentimentPi.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['label','count'])
    for x in s1:
        file_writer.writerow(x)

'''-----------------------------------------------------------------------------'''
'''------------------ CSV Output of Phrase / Root Cause---------------------------------'''
'''-----------------------------------------------------------------------------'''
d = []
d = Counter(All_Phrase).most_common(200)

#original: with open('result\\PhraseTheme.csv', 'wb') as test_file:
with open('result\\PhraseTheme.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    #original file_writer.writerow(['Theme','Frequency'])
    file_writer.writerow(['Theme','Frequency'])
    for x in d:
        file_writer.writerow(x)

'''---------------------------------------------------'''
d = []
d = Counter(Pos_Phrase).most_common(200)
#original with open('result\\PhraseTheme_Pos.csv', 'wb') as test_file:
with open('result\\PhraseTheme_Pos.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Theme','Frequency'])
    for x in d:
        file_writer.writerow(x)

'''---------------------------------------------------'''
d = []
d = Counter(Neg_Phrase).most_common(200)

#original with open('result\\PhraseTheme_Neg.csv', 'wb') as test_file:
with open('result\\PhraseTheme_Neg.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Theme','Frequency'])
    for x in d:
        file_writer.writerow(x)

'''---------------------------------------------------'''
d = []
d = Counter(Neu_Phrase).most_common(200)
#original with open('result\\PhraseTheme_Ntrl.csv', 'wb') as test_file:
with open('result\\PhraseTheme_Ntrl.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Theme','Frequency'])
    for x in d:
        file_writer.writerow(x)

'''-----------------------------------------------------------------------------'''
nn = []
nn = Counter(All_Noun).most_common(200)
#original with open('result\\TalkingAbout_Noun.csv', 'wb') as test_file:
with open('result\\TalkingAbout_Noun.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Phrase','Frequency'])
    for x in nn:
        file_writer.writerow(x)

'''-----------------------------------------------------------------------------'''
vb = []
vb = Counter(All_Verb).most_common(200)
#original with open('result\\TalkingWhat_Verb.csv', 'wb') as test_file:
with open('result\\TalkingWhat_Verb.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Phrase','Frequency'])
    for x in vb:
        file_writer.writerow(x)

'''-----------------------------------------------------------------------------'''
b = []
b = Counter(WrdCrps).most_common(900000)
#original with open('result\\TermFreqMatrix1.csv', 'wb') as test_file:
with open('result\\TermFreqMatrix1.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Word','Frequency'])
    for x in b:
        file_writer.writerow(x)
'''-----------------------------------------------------------------------------'''
b = []
b = Counter(All_Positive).most_common(2000)
#original with open('result\\Freq_Positive_Words.csv', 'wb') as test_file:
with open('result\\Freq_Positive_Words.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Word','Frequency'])
    for x in b:
        file_writer.writerow(x)

'''-----------------------------------------------------------------------------'''
b = []
b = Counter(All_Negative).most_common(2000)
#original with open('result\\Freq_Negative_Words.csv', 'wb') as test_file:
with open('result\\Freq_Negative_Words.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Word','Frequency'])
    for x in b:
        file_writer.writerow(x)
'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
c = []
c = Counter(WrdCrps_cleaned).most_common(2000)
#with open('result\\TermFreqMatrix_Cleaned.csv', 'wb') as test_file:
with open('result\\TermFreqMatrix_Cleaned.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Word','Frequency'])
    for x in c:
        file_writer.writerow(x)


'''----------------------------------------------------------------------------------------'''
'''------------------- Associations using NGram ----------------------------------------------'''
'''---------------- Not Using as Not Working properly -------------------------------------'''

corpus = WrdCrps

#variables PosWords and NegWords were list objects created by fetch from database
def Pngrams(input, n):
    output = []
    for i in range(len(input)-n+1):
        if input[i] in PosWords:
            output.append(input[i:i+n])
        if input[i+n-1] in PosWords:
            output.append(input[i:i+n])
    return output

def Nngrams(input, n):
    output2 = []
    for i in range(len(input)-n+1):
        if input[i] in NegWords:
            output2.append(input[i:i+n])
        if input[i+n-1] in NegWords:
            output2.append(input[i:i+n])
    return output2

def FnRemovePNToken(string):
    ngram=[]
    for token in string:
        if token not in PosWords:
            if token not in NegWords:
                ngram.append(token)
    return ngram

#Next word
pGram2 = Pngrams(corpus, 2)
nGram2 = Nngrams(corpus, 2)

#Next to next word
pGram3 = Pngrams(corpus, 3)
nGram3 = Nngrams(corpus, 3)




'''---------------WRITE THE NGRAM TO CSV -----------------------------'''
#original with open('result\\PositiveWordAssociations1.csv', 'wb') as test_file:
with open('result\\PositiveWordAssociations1.csv', 'w' , newline='' ) as test_file:
    file_writer = csv.writer(test_file)
    #x = str()
    file_writer.writerow(['Word','NextWord'])
    for x in pGram2:
        file_writer.writerow(x)
#original with open('result\\NegativeWordAssociations1.csv', 'wb') as test_file:
with open('result\\NegativeWordAssociations1.csv', 'w' , newline='' ) as test_file:
    file_writer = csv.writer(test_file)
    file_writer.writerow(['Word','NextWord'])
    for x in nGram2:
        file_writer.writerow(x)


'''-----------After removal of Positive & negative Word from association -----------'''
# 2 Gram
pGram2 = reduce(lambda x,y: x+y,pGram2)   # Flat the list
pGram2 = FnRemovePNToken(pGram2)          # After removing Positive Word from the list

nGram2 = reduce(lambda x,y: x+y,nGram2)   # Flat the list
nGram2 = FnRemovePNToken(nGram2)          # After removing Positive Word from the list

pGram2 = Counter(pGram2).most_common(200)
nGram2 = Counter(pGram2).most_common(200)

# 3 Gram
pGram3 = reduce(lambda x,y: x+y,pGram3)   # Flat the list
pGram3 = FnRemovePNToken(pGram3)          # After removing Positive Word from the list

nGram3 = reduce(lambda x,y: x+y,nGram3)   # Flat the list
nGram3 = FnRemovePNToken(nGram3)          # After removing Positive Word from the list

pGram3 = Counter(pGram3).most_common(200)
nGram3 = Counter(pGram3).most_common(200)


'''' ----------- Write nGram Frequency to csv ---------------- '''

#original with open('result\\Next2PosWrd1.csv', 'wb') as test_file:
with open('result\\Next2PosWrd1.csv', 'w' , newline='' ) as test_file:
    file_writer = csv.writer(test_file)
    file_writer.writerow(['Word','Freq'])
    for x in pGram2:
        file_writer.writerow(x)

#original with open('result\\Next2NegWrd1.csv', 'wb') as test_file:
with open('result\\Next2NegWrd1.csv', 'w' , newline='' ) as test_file:
    file_writer = csv.writer(test_file)
    file_writer.writerow(['Word','Freq'])
    for x in nGram2:
        file_writer.writerow(x)

# After 3rd position
#original with open('result\\Next3PosWrd1.csv', 'wb') as test_file:
with open('result\\Next3PosWrd1.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    file_writer.writerow(['Word','Freq'])
    for x in pGram3:
        file_writer.writerow(x)

#original with open('result\\Next3NegWrd1.csv', 'wb') as test_file:
with open('result\\Next3NegWrd1.csv', 'w' , newline='') as test_file:
    file_writer = csv.writer(test_file)
    file_writer.writerow(['Word','Freq'])
    for x in nGram3:
        file_writer.writerow(x)


'''---------------------------------------------------------------'''
'''------------- Word Cloud ----------------------------------------'''
'''---------------------------------------------------------------'''


'''---- All Phrase -----------------------------'''

#text1 = str(All_Phrase)
#wordcloud1 = WordCloud(font_path='Data/includes/Attic.ttf', background_color='white', width=1200, height=1000).generate(text1)
#plt.imshow(wordcloud1)
#plt.axis('off')
#wordcloud1.to_file("result\\Phrase_WC.png")

elapsed_time = time.clock() - start_time
print("Completed Processing with CSV files output")
print ("Time elapsed: {} seconds".format(elapsed_time))

"""
