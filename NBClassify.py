__author__ = 'Suman_pa'
import collections

from nltk.corpus import movie_reviews
from featx import label_feats_from_corpus, bag_of_words, split_label_feats
#from tabulate import tabulate
from nltk.classify import NaiveBayesClassifier

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats

def split_label_feats(lfeats, split=.70):
    train_feats = []
    test_feats = []
    #original: for label, feats in lfeats.iteritems():
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats


movie_reviews.categories()

lfeats = label_feats_from_corpus(movie_reviews)

#print lfeats
#print lfeats.keys()
train_feats, test_feats = split_label_feats(lfeats)
#train_feats = lfeats
#print len(train_feats)
#print len(test_feats)
#print tabulate(train_feats[0:2], tablefmt="grid")

#---------------------------------------------------------------------
nb_classifier = NaiveBayesClassifier.train(train_feats)
#print nb_classifier.labels()



#test = bag_of_words(['hello', 'the', 'movie', 'was', 'not', 'awesome', 'though', 'I', 'did','not', 'like', 'that'])
#print test

#print nb_classifier.classify(test)










