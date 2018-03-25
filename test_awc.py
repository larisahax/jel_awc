import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from awc import AWC
#from text_error_overlap import overlap_error
import measures
from abstracts_loader import *
#from AWC_NIPS_A_new_idea_fast_3_my_library_draft import AWC


def test_plateau(X, lamdas, dataset, n_0=-1, dim=None, true_weights=None,  dist_matrix=None, true_clusters=None, filename=None, lamdas_to_show=None, dir_name='results/'):
    n = len(X)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if filename == None:
        filename = dir_name + dataset + ',n0=' + str(n_0) + ',dim=' + str(dim) + '.awc'
    if os.path.isfile(filename):
        [lamdas_computed, weights_computed, clustering_computed] = pickle.load(open(filename))
    else:
        print 'dirname:',  dir_name, filename
        lamdas_computed, weights_computed, clustering_computed = [], [], []
    
    awc_obj = AWC(n_neigh=n_0, effective_dim=dim, n_outliers=0, discrete=False, speed=1.)
    for l in lamdas:
        print 'lambda= ', l
        if l in lamdas_computed:
            weights = weights_computed[lamdas_computed.index(l)]
        else:    
            weights = awc_obj.awc(l, X)
            clustering = awc_obj.get_clusters()
            #clustering = clustering_from_weights(weights)
            #weights, clustering = AWC(X, l, dmatrix=dist_matrix, n_neigh=n_0, dim=dim, clustering=True, tclusters=true_clusters, true_weights=true_weights)
            lamdas_computed.append(l)
            weights = weights > 0.5
            weights_computed.append(weights)
            clustering_computed.append(clustering)
            pickle.dump([lamdas_computed, weights_computed, clustering_computed], open(filename,'w+'))

    print 'computed lambdas: ', lamdas_computed
    if lamdas_to_show != None:
        for i in reversed(xrange(len(lamdas_computed))):
            if lamdas_computed[i] < lamdas_to_show[0] or lamdas_computed[i] > lamdas_to_show[1]:
                del lamdas_computed[i]
                del weights_computed[i]
                del clustering_computed[i]
    sorting_order = np.argsort(lamdas_computed)
    lamdas_computed_t = np.asarray(lamdas_computed)
    weights_summed = [np.sum(weights_computed[i]) * 1. / (n**2) for i in xrange(len(lamdas_computed))]
    weights_summed = np.asarray(weights_summed)
    
    if true_weights.any() != None:
        e = np.asarray([measures.get_error(weights_computed[i], true_weights) for i in xrange(len(lamdas_computed))])
        e_s = np.asarray([measures.get_error(weights_computed[i], true_weights, True)[0] for i in xrange(len(lamdas_computed))])
        e_p = np.asarray([measures.get_error(weights_computed[i], true_weights,True)[1] for i in xrange(len(lamdas_computed))])
        print "min error(best lamda): %s(%s)  " % (round(np.min(e), 3), lamdas_computed_t[e.tolist().index(np.min(e))])

        
    nmi = np.asarray([measures.NMI(clustering_computed[i], true_clusters, n) for i in xrange(len(lamdas_computed))])
    #nmi = np.asarray([measures.correct_NMI(true_clusters, clustering_computed[i], len(X), 'nmi') for i in xrange(len(lamdas_computed))])
    #overlaperror = np.asarray([overlap_error(clustering_computed[i]) for i in xrange(len(lamdas_computed))])
    fs = np.asarray([measures.Fscore(clustering_computed[i], true_clusters, n) for i in xrange(len(lamdas_computed))])
    arand = np.asarray([measures.AdRand(clustering_computed[i], true_clusters, n) for i in xrange(len(lamdas_computed))])
    pur = np.asarray([measures.Purity(clustering_computed[i], true_clusters, n) for i in xrange(len(lamdas_computed))])
    entropy = np.asarray([measures.Entropy(clustering_computed[i], true_clusters, n) for i in xrange(len(lamdas_computed))])
    
    #for i in xrange(len(overlaperror)):
    #    if np.isnan(overlaperror[i]):
    #        overlaperror[i] = 0
    #print "max overlap error(best lamda): %s(%s)  " % (round(np.max(overlaperror), 3), lamdas_computed_t[overlaperror.index(np.max(overlaperror))])

    print "max ARI(best lamda): %s(%s)  " % (round(np.max(arand), 3), lamdas_computed_t[arand.tolist().index(np.max(arand))])
    print "max NMI(best lamda): %s(%s)  " % (round(np.max(nmi),3), lamdas_computed_t[nmi.tolist().index(np.max(nmi))])
    print "max F1(best lamda): %s(%s)  " % (round(np.max(fs),3), lamdas_computed_t[fs.tolist().index(np.max(fs))])
    print "max Purity(best lamda): %s(%s)  " % (round(np.max(pur),3), lamdas_computed_t[pur.tolist().index(np.max(pur))])
    print "min Entropy(best lamda): %s(%s)  " % (round(np.min(entropy),3), lamdas_computed_t[entropy.tolist().index(np.min(entropy))])

    
    #plt.plot(lamdas_computed_t[sorting_order], e[sorting_order], 'k', label='error',linewidth=3, dashes=[10, 4])#), markerfacecolor='grey',markeredgecolor='k')
    #plt.plot(lamdas_computed_t[sorting_order], e_p[sorting_order], ':k', label=r'$e_p$',linewidth=2, dashes=[2,2])
    #plt.plot(lamdas_computed_t[sorting_order], e_s[sorting_order], 'k', label=r'$e_s$',linewidth=2, dashes=[8, 4, 2, 4, 2, 4])
    #plt.plot(lamdas_computed_t[sorting_order], overlaperror[sorting_order],'b', label='NMI', linewidth=2, dashes=[8, 4, 2, 4, 2, 4])

    plt.plot(lamdas_computed_t[sorting_order], nmi[sorting_order], 'b', label='NMI', linewidth=3)
    plt.plot(lamdas_computed_t[sorting_order], fs[sorting_order], 'g', label='F', linewidth=2)# dashes=[8, 4, 2, 4, 2, 4])
    plt.plot(lamdas_computed_t[sorting_order], arand[sorting_order], 'r', label='ARI', linewidth=3)#, dashes=[8, 4, 2, 4, 2, 4])
    plt.plot(lamdas_computed_t[sorting_order], pur[sorting_order], 'darkorange', label='Purity', linewidth=3, dashes=[2,2])
    plt.plot(lamdas_computed_t[sorting_order], entropy[sorting_order], 'c', label='Entropy', linewidth=3, dashes=[2,2])
    

    plt.plot(lamdas_computed_t[sorting_order], weights_summed[sorting_order], 'k', label=r'$\sum w_{ij}$',linewidth=3)

    plt.legend(loc=2, fontsize=16)
    plt.xlabel(r'$\lambda$', size=20)
    #plt.ylabel(r'$\sum w_{ij}$', size=17)
    #plt.title('AWC')
    #plt.title('abstracts_first_jel')
    plt.grid(b=True, which='major', color='k', linestyle='--')
    #plt.grid(b=True, which='minor', color='r', linestyle='-', linewidth=0)
    #plt.xticks(np.arange(min(lamdas_computed_t[sorting_order]), max(lamdas_computed_t[sorting_order])+1, 2.0))
    plt.show()
    return 0


def get_tweights(X, true_clusters):
    n = len(X)
    true_weights = np.zeros((n, n))
    for c in true_clusters:
        for i in c:
            for j in c:
                true_weights[i,j] = 1

    return true_weights

def main():
    if os.path.isfile('preproc_doc_jel.lmd') == 0:
        save_preproc_doc_jel()
    X, true_clusters = pickle.load(open('preproc_doc_jel.lmd', 'r'))
    print true_clusters
    #print measures.correct_NMI(true_clusters, true_clusters, len(X), 'nmi')
    #return
    n_0=40
    dim=2
    dataset = 'slow_abstract_first_jel'
    print np.shape(X)
    #lamdas = np.linspace(0., 10., 31, 1)
    lamdas=[1.]
    true_weights = get_tweights(X, true_clusters)
    test_plateau(X, lamdas, dataset,  n_0, dim, true_weights=true_weights, true_clusters=true_clusters, lamdas_to_show=[-0.1, 1.5])

    
if __name__ == '__main__':
    main()