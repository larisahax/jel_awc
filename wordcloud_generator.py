from wordcloud import WordCloud
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

def count_idfs(abstracts):
    #for k in xrange(len(abstracts)):
    #    abstracts[k] = [word for word in abstracts[k].split() ]
    word_idfs = dict()
    words = set(x for l in abstracts for x in l)
    words = list(words)
    print words
    for word in words:
        n_word = 0
        for a in abstracts:
            if word in a:
                n_word += 1
        idf = np.log((1 + len(abstracts)) * 1.0/ (1 + n_word)) + 1
        word_idfs[word] = idf
    for w in word_idfs.keys():
        print w, word_idfs[w]
    pickle.dump(word_idfs, open('word_idfs','w+'))
    
def color_func(word, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    
    word_idfs = pickle.load(open('word_idfs','r'))
    print word_idfs[word]
    values = word_idfs.values()
    #color = int(256 * word_idfs[word] / (np.max(values) - np.min(values)))
    #return color
    a = 0
    b = 90
    t = -1
    color = int(100 * word_idfs[word] / (np.max(values) - 0))
    if t == 1:
        color = int(color / 100. * (b-a) + a)
    if t == -1:
        color = int(- color / 100. * (b-a) + b)
    return "hsl(%d, 80%%, %d%%)" % (color / 2, color)


def save_clusters(abstracts, answer_clustering):
    for i in xrange(len(answer_clustering)):
        cl = [abstracts[j] for j in answer_clustering[i]]
        with open('cluster' + str(i) +'.txt', "w") as text_file:
            for t in cl:
                text_file.write(t + '\n')

def generate_wordcloud(cluster_file='cluster1.txt'):
    text = open(cluster_file).read()
    wordcloud = WordCloud(background_color="white", max_font_size=80, min_font_size=13, width=800, height=400, color_func=color_func, random_state=5).generate(text)
    rcParams['figure.figsize'] = 10, 10
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
def main():
    X = pickle.load(open('chinese_abstracts.lmd','r'))[2]    
    count_idfs(X)
    answer_clustering = pickle.load(open('chinese_l=0.3.lmd','r'))[1]
    save_clusters(X, answer_clustering)
    generate_wordcloud('chinese_cluster3.txt')     
    
    
def count_jel_percent():
    pcodes, jels, X = pickle.load(open('chinese_abstracts.lmd','r'))
    answer_clustering = pickle.load(open('chinese_l=0.3.lmd','r'))[1]   
    print answer_clustering 
    '''cl_0 = answer_clustering[0]
    print cl_0
    j1, j2 = 'C', 'G'
    cont = 0
    for i in cl_0:
        if j1 in jels[i] or j2 in jels[i]:
            cont += 1 
    percent = cont * 100 / len(cl_0)
    print percent '''  
    
    '''cl_1 = answer_clustering[1]
    print cl_1
    j1 = 'C'
    cont = 0
    for i in cl_1:
        print jels[i]
        if j1 in jels[i]:
            cont += 1 
    percent = cont * 100 / len(cl_1)
    print percent '''
    
    cl_2 = answer_clustering[3]
    print cl_2
    j1 = 'O'
    j2 = 'L'
    cont = 0
    for i in cl_2:
        print jels[i]
        if j1 in jels[i] and j2 in jels[i]:
            cont += 1 
    percent = cont * 100 / len(cl_2)
    print percent 


 
def table_cluster(answer_clustering, true_clustering, names_true=None):
    k_a = len(answer_clustering)
    k_t = len(true_clustering)
    table = np.zeros((k_a, k_t))
    for i in xrange(k_a):
        for j in xrange(k_t):
            table[i, j] = len(list(set(answer_clustering[i]) & set(true_clustering[j])))
            table[i, j] /= 1. * len(true_clustering[j])
            table[i,j] = 1. * int(table[i,j] * 100) / 100.
            
    from prettytable import PrettyTable
    names = [''] + [str(i+1) for i in xrange(len(true_clustering))]
    names = ['', 'C', 'G', 'E', 'D', 'F', 'Q', 'R', 'H', 'N', 'J', 'K', 'L', 'O', 'A', 'M', 'I', 'B']
    t = PrettyTable(names)
    t.add_row(['n_i'] + ['(' + str(len(true_clustering[i]))+ ')' for i in xrange(len(true_clustering))])
    for i in xrange(len(answer_clustering)):
        e = list(table[i, :])
        for e_i in range(len(e)):
            if e[e_i] < 0.1:
                e[e_i] = ''
        t.add_row(['AWC_' + str(i+1) + ' (' + str(len(answer_clustering[i]))+ ')'] + e)
    #table = np.round(table, 3)
    #return    
    
    print t
    return table       
        
if __name__ == '__main__':
    main()    
 