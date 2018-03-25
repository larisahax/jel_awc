import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

import string

def convert_to_numbers(Y):
    clusters = dict()
    result = []
    n = 0
    for y in Y:
        if y in clusters.keys():
            result.append(clusters.get(y))
        else:
            clusters[y] = n
            result.append(n)
            n = n+1
    return result


def load_jel_abstracts():
    '''Load from the file documents with their JEL codes. Documents are split into stemmed words.'''
    stemmer = EnglishStemmer()
    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/jel_abstracts.txt') as generate_wordcloud:
            X = generate_wordcloud.readlines()
    del X[0]
    pcodes = []
    jels = []
    abstracts = []
    for i in xrange(len(X)):
        line = X[i].split('"')
        line = [line[i] for i in xrange(len(line)) if line[i] != '' and line[i] != ' '] 
        pcodes.append(line[1])
        line = line[2:-1]
        if len(line) == 1:
            jel = ''
            doc = line[0]
            #print string
        else:
            jel = line[0]
            doc = line[1]
        
        jel = ''.join([i for i in jel if not i.isdigit() and i not in set(string.punctuation)])
        
        doc = [word for word in doc.split() if word not in stopwords.words("english")]
        doc = [stemmer.stem(doc[i]) for i in xrange(len(doc))]
        
        jels.append(jel)
        abstracts.append(doc)  
        
    #print jels
    print 'jels shape', np.shape(jels)
    print 'X', np.shape(abstracts)
    print 'jels', jels
    return pcodes, jels, abstracts  

def containsNonAscii(s):
    return any(ord(i)>127 for i in s)

def docs2vector(X):  
    '''Convert text docs to vector representations'''  
    dict = list(set(x for l in X for x in l))
    #dict = dict[1:]
    data = np.zeros((len(X), len(dict)))
    for i in xrange(len(X)):
        for j in xrange(len(dict)):
            data[i, j] = X[i].count(dict[j])
    
    li = []
    for k in xrange(np.size(data, 1)):
        if len(data[:, k][data[:,k] != 0]) > 2:
            li.append(k)
    data = data[:, li]

    X = TfidfTransformer().fit_transform(data)
    X = X.toarray()
    print np.shape(X)
    print X[0]
    #pickle.dump(X, open('abstracts.lmd','w+'))
    return X

def get_tclusters(Y):
    n = len(Y)
    true_clusters = []
    cluster_marker = []
    for i in xrange(n):
        if Y[i] in cluster_marker:
            true_clusters[cluster_marker.index(Y[i])].append(i)
        else:
            cluster_marker.append(Y[i])
            true_clusters.append([i])
    return true_clusters

def first_jel_as_tcluster(jels):
    '''Assign the first JEL code as a true clustering label.'''
    jels = [jels[i].split() for i in xrange(len(jels))]
    jels = [jels[i][0] for i in xrange(len(jels))]
    print set(jels)
    #covert JEL codes to numerical labels
    Y = convert_to_numbers(jels)
    tclusters = get_tclusters(Y)
    print 'len tclus', len(tclusters), 'tclusters', tclusters
    return tclusters


def save_preproc_doc_jel():
    '''save the preprocessed docs with the first JEL codes as true cluster labels to a file'''
    pcodes, jels, abstracts = load_jel_abstracts()
    X = docs2vector(abstracts)
    tclusters = first_jel_as_tcluster(jels)
    pickle.dump([X, tclusters], open('preproc_doc_jel.lmd','w+'))
    
def main():
    load_jel_abstracts()
    #save_preproc_doc_jel()
    #X, true_clusters = pickle.load(open('preproc_doc_jel.lmd', 'r'))
    return

    
if __name__ == '__main__':
    main()
