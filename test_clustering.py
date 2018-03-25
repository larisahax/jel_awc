from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import pickle
import random
from abstracts_loader import *
from measures import *
import matplotlib.pyplot as plt

def run_kmeans(X, true_clusters):
    n_clus = range(2, 20)
    #n_clus = [3]
    rindexs = []
    nmis = []
    errors = []
    fs = []
    purs = []
    entropys = []
    n = len(X)
    overlaperrors = []
    #pca = decomposition.FastICA(n_components=2)
    #pca.fit(X)
    #X = pca.transform(X)
    for n_c in n_clus:
        rindex = 0
        nmi = 0
        error = 1
        overlaperror = 0
        f_measure = 0
        pur = 0
        entr = 1000
        for p in range(50):
            random_state = random.randint(1, 500)
            labels = KMeans(n_c, random_state=random_state).fit_predict(X)
            labels = convert_to_numbers(labels)
            clusters = get_tclusters(labels)
            #if AdRand(clusters, true_clusters, len(X)) > rindex:
            #    best_weights = weights
            rindex = max(rindex, AdRand(clusters, true_clusters, n))
            nmi = max(nmi, NMI(clusters, true_clusters, n))
            #nmi = max(nmi, correct_NMI(clusters, true_clusters, len(X), 'nmi'))
            f_measure = max(f_measure, Fscore(clusters, true_clusters, n))
            pur = max(pur, Purity(clusters, true_clusters, n))
            entr = min(entr, Entropy(clusters, true_clusters, n))
            #overlaperror = max(overlaperror, overlap_error(clusters))
            #error = min(error, get_error(weights, true_weights, separate_errors))
        #overlaperrors.append(overlaperror)
        rindexs.append(rindex)
        nmis.append(nmi)
        fs.append(f_measure)
        purs.append(pur)
        entropys.append(entr)
    best_rindex = np.max(rindexs)
    best_nmi = np.max(nmis)
    #best_error = min(errors)        
    #print 'best_error', best_error
    print 'best_nmi', best_nmi
    print 'best_rindex', best_rindex
    
    #plt.plot(n_clus, overlaperrors, 'k', label='NMI', linewidth=3)
    #plt.plot(n_clus, errors, 'k', label='error', linewidth=3)
    plt.plot(n_clus, nmis, 'b', label='NMI', linewidth=3)
    plt.plot(n_clus, rindexs, 'r', label='ARI', linewidth=3)
    plt.plot(n_clus, fs, 'g', label='F', linewidth=3)
    plt.plot(n_clus, purs, 'darkorange', label='Purity', linewidth=3, dashes=[2,2])
    plt.plot(n_clus, entropys, 'c', label='Entropy', linewidth=3, dashes=[2,2])  

    plt.legend(fontsize=16)
    plt.xlabel('K', size=20)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    #plt.title('K-means')
    plt.show()
    return
    labels = KMeans(5).fit_predict(X)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                weights[i, j] = 1
    #draw_step(weights, X, clustering=True)
    #draw(X, labels, true_clusters, weights, true_weights) 

import subprocess
    
def get_weights_from_labels(labels):
    n = len(labels)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                weights[i, j] = 1
    return weights
    
def cluto_clustering(dataset, k, weights_flag = False):
    #path = os.path.dirname(os.path.abspath(__file__)) + '/cluto/'
    path = '/Users/larisa/Downloads/cluto-2.1.2/Matrices/'
    labels = np.genfromtxt(path + 'abstracts.mat.clustering.' + str(k))
    clustering = get_tclusters(labels)
    if weights_flag == True:
        n = len(labels)
        weights = get_weights_from_labels(labels)
        return clustering, weights
    return clustering

def run_cluto(X, true_clusters):
    dataset = 'abstracts'
    #true_weights = get_tweights(X, true_clusters)
    k = len(true_clusters)
    print 'K = ', k
    nmis = []
    path = '/Users/larisa/Downloads/cluto-2.1.2/Darwin-i386/'
    os.chdir(path)    
    #k -= 1
    
    k_list = range(2, 20)
    max_rindexs = []
    max_nmis = []
    min_errors = []
    max_overlaps = []
    min_rindexs = []
    min_nmis = []
    max_errors = []
    
    fs = []
    purs = []
    entropys = []
    
    best_weights = None
    for k in k_list:
        max_rindex = 0
        max_nmi = 0
        min_error = 1
        max_overlap = 0
        
        min_rindex = 1
        min_nmi = 1
        max_error = 0
        n = len(X)
        for i in xrange(50):
            f_measure = 0
            pur = 0
            entr = 1000
            #seed = random.randint(1, 10) 
            #print seed
            command = ['./vcluster','-rclassfile=../Matrices/'+ dataset + '.mat.rclass', 
                       '-clabelfile=../Matrices/' + dataset + '.mat.clabel', 
                       '../Matrices/' +dataset + '.mat', '-seed=' + str(i), str(k)]
            print subprocess.check_output(command)
            clustering, weights = cluto_clustering(dataset, k, True)
            if AdRand(clustering, true_clusters, len(X)) > max_rindex:
                best_weights = weights
            max_nmi = max(NMI(clustering, true_clusters, n), max_nmi)
            max_rindex = max(AdRand(clustering, true_clusters, n), max_rindex)
            #min_error = min(min_error, get_error(weights, true_weights, 0))
            #max_overlap = max(overlap_error(clustering), max_overlap)
            
            min_nmi = min(NMI(clustering, true_clusters, n), min_nmi)
            min_rindex = min(AdRand(clustering, true_clusters, n), min_rindex)
            #max_error = max(max_error, get_error(weights, true_weights, 0))
            
            f_measure = max(f_measure, Fscore(clustering, true_clusters, n))
            pur = max(pur, Purity(clustering, true_clusters, n))
            entr = min(entr, Entropy(clustering, true_clusters, n))
        
        max_rindexs.append(max_rindex)
        max_nmis.append(max_nmi)
        min_errors.append(min_error)
        max_overlaps.append(max_overlap)
        
        min_rindexs.append(min_rindex)
        min_nmis.append(min_nmi)
        max_errors.append(max_error)
        
        fs.append(f_measure)
        purs.append(pur)
        entropys.append(entr)
    
    best_rindex = np.max(max_rindexs)
    best_nmi = np.max(max_nmis)
    best_error = min(min_errors)        
    print 'best_error', best_error
    print 'best_nmi', best_nmi
    print 'best_rindex', best_rindex
    print 'best overlap error', np.max(max_overlaps)
    print np.shape(best_weights)
    #draw_step(best_weights, X, true_clusters=true_clusters, clustering=True, true_weights=true_weights)

    #plt.plot(k_list, min_errors, 'k', label='error min', linewidth=3)
    plt.plot(k_list, max_nmis, 'b', label='NMI', linewidth=3)
    plt.plot(k_list, max_rindexs, 'r', label='ARI', linewidth=3)
    #plt.plot(k_list, max_overlaps,'k', label='NMI', linewidth=3)
    plt.plot(k_list, fs, 'g', label='F', linewidth=3)
    plt.plot(k_list, purs, 'darkorange', label='Purity', linewidth=3, dashes=[2,2])
    plt.plot(k_list, entropys, 'c', label='Entropy', linewidth=3, dashes=[2,2])  
    
    plt.legend(fontsize=16)
    plt.xlabel('K', size=20)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    #plt.ylabel(r'$\sum w_{ij}$', size=17)
    #plt.title('Cluto')
    plt.show()

    return



  
def main():
    if os.path.isfile('preproc_doc_jel.lmd') == 0:
        save_preproc_doc_jel()
    X, true_clusters = pickle.load(open('preproc_doc_jel.lmd', 'r'))
    #run_cluto(X, true_clusters)
    run_kmeans(X, true_clusters)

if __name__ == '__main__':
    main()