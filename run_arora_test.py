__author__ = 'Bryce Woodworth'

import os
import time
import numpy
import pickle
from preprocessor import *
from improved_arora import *


def preprocess(vocab_size):
    # First do some preprocessing and clean up the dataset
    base_dir = 'datasets/SentenceCorpus/unlabeled_articles'
    stop_file = 'datasets/SentenceCorpus/word_lists/stopwords.txt'
    clean_dir = 'datasets/SentenceCorpus/cleaned'
    prune_dir= 'datasets/SentenceCorpus/pruned'

    with open(stop_file) as stopfile:
        stops = [stop.strip() for stop in stopfile]

    dirs = [os.path.join(base_dir, dir) for dir in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, dir)) and not dir.startswith('.')]

    allfiles = [os.path.join(direct, file) for direct in dirs for file in os.listdir(direct)
                if os.path.isfile(os.path.join(direct, file)) and not file.startswith('.')]

    cleandirs = {}
    for dir in dirs:
        files = [os.path.join(dir, file) for file in os.listdir(dir)
                 if os.path.isfile(os.path.join(dir, file)) and not file.startswith('.')]
        cleandir = os.path.join(clean_dir, os.path.basename(dir))
        cleandirs[dir] = cleandir
        stop_remover(files, stops, cleandir)

    cleanfiles = [os.path.join(direct, file) for (_, direct) in cleandirs.items() for file in os.listdir(direct)
                  if os.path.isfile(os.path.join(direct, file)) and not file.startswith('.')]

    vocab = find_most_common(cleanfiles, vocab_size)

    for dir in dirs:
        cleandir = cleandirs[dir]
        files = [os.path.join(cleandir, file) for file in os.listdir(cleandir)
                 if os.path.isfile(os.path.join(cleandir, file)) and not file.startswith('.')]
        prunedir = os.path.join(prune_dir, os.path.basename(dir))
        word_filter(files, vocab, prunedir)

    # Next we get the Q matrix from the data
    dirs = [os.path.join(prune_dir, dir) for dir in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, dir)) and not dir.startswith('.')]
    allfiles = [os.path.join(direct, file) for direct in dirs for file in os.listdir(direct)
                if os.path.isfile(os.path.join(direct, file)) and not file.startswith('.')]

    H = numpy.zeros((vocab_size, len(allfiles)))  # The term-by-document matrix
    for f in range(len(allfiles)):
        with open(allfiles[f]) as file:
            for line in file:
                for word in line.split():
                    H[vocab.index(word)][f] += 1

    Q = construct_Q(H)

    priors = [0.0] * Q.shape[0]
    for i in range(len(priors)):
        priors[i] = numpy.sum(Q[i])
    priors = priors / sum(priors)

    Qbar = numpy.empty_like(Q)
    for row in range(Q.shape[0]):
        n = numpy.sum(Q[row])
        if n <= 0.0:
            print("invalid sum of %d at row %d" % (n, row))
        Qbar[row] = Q[row] / n

    return vocab, priors, Qbar, H


def _recalculate(num_topics, vocab_size):
    """Simply runs the algorithm to recalculate the desired values"""
    print("\nBeginning preprocessing.")
    start = time.time()
    vocab, priors, Q, H = preprocess(vocab_size)
    print("Preprocessed and found Q in %d seconds." % (time.time() - start))
    inter = time.time()
    S = anchor_selection(Q, num_topics)
    print("Found anchor words in %d seconds." % (time.time() - inter))
    A, C = recover(Q, S, priors)
    print("Recovered A in %d seconds." % (time.time() - start))
    return vocab, priors, Q, H, S, A, C


def _recalculate2(num_topics, vocab_size):
    """Simply runs the algorithm to recalculate the desired values"""
    print("\nBeginning preprocessing.")
    start = time.time()
    vocab, priors, Q, H = preprocess2(vocab_size)
    print("Preprocessed and found Q in %d seconds." % (time.time() - start))
    inter = time.time()
    S = anchor_selection(Q, num_topics)
    print("Found anchor words in %d seconds." % (time.time() - inter))
    A, C = recover(Q, S, priors)
    print("Recovered A in %d seconds." % (time.time() - start))
    return vocab, priors, Q, H, S, A, C


def preprocess2(vocab_size):
    path = "datasets/NIPS/"
    stopfile = "anchor-word-recovery/stopwords.txt"
    return word_bag_reduce(path + "docword.nips.txt", path + "vocab.nips.txt", "/usr/share/dict/words", vocab_size, stopwords=stopfile)


if __name__ == '__main__':
    try:
        num_topics_nips, vocab_size_nips, vocab_nips, priors_nips, Q_nips, H_nips, S_nips, A_nips, C_nips = pickle.load(open('saved/NIPS.p', 'rb'))
        nips_saved = True
    except FileNotFoundError:
        nips_saved = False

    try:
        num_topics_tri, vocab_size_tri, vocab_tri, priors_tri, Q_tri, H_tri, S_tri, A_tri, C_tri = pickle.load(open('saved/tri.p', 'rb'))
        tri_saved = True
    except FileNotFoundError:
        tri_saved = False

    save_loaded = False
    while True:
        dataset = input('Please select a dataset (default is 0).\n'
                        '0 - Introductions and abstracts of 300 papers each from '
                        'machine learning, computational biology, and psychology (tri-journal).\n'
                        '1 - Full data from 1500 NIPS papers.\n')
        if dataset == '':
            dataset = '0'
        if dataset == '0':
            if tri_saved:
                response = input("Previous tri-journal save found - vocabulary size: %d, number of topics: %d\n"
                                 "Would you like to use this save? (y/N):\n" % (vocab_size_tri, num_topics_tri))
                response = response.lower()
                if response == 'y' or response == 'yes':
                    num_topics, vocab_size, vocab, priors, Q, H, S, A, C = num_topics_tri, vocab_size_tri, vocab_tri, \
                                                                           priors_tri, Q_tri, H_tri, S_tri, A_tri, C_tri
                    save_loaded = True
            break

        elif dataset == '1':
            if nips_saved:
                response = input("Previous nips-journal save found - vocabulary size: %d, number of topics: %d\n"
                                 "Would you like to use this save? (y/N):\n" % (vocab_size_nips, num_topics_nips))
                response = response.lower()
                if response == 'y' or response == 'yes':
                    num_topics, vocab_size, vocab, priors, Q, H, S, A, C = num_topics_nips, vocab_size_nips, vocab_nips,\
                                                                           priors_nips, Q_nips, H_nips, S_nips, A_nips, C_nips
                    save_loaded = True
            break

        else:
            print('invalid input\n')

    while not save_loaded:
        num_topics = input('Please select how many topics you want to start out with (default is 20): ')
        if num_topics == '':
            num_topics = 20
            break
        else:
            try:
                num_topics = int(num_topics)
                break
            except ValueError:
                print("invalid input")

    while not save_loaded:
        vocab_size = input('Please select the size of your vocabulary (default is 1000): ')
        if vocab_size == '':
            vocab_size = 1000
            break
        else:
            try:
                vocab_size = int(vocab_size)
                break
            except ValueError:
                print("invalid input")

    if not save_loaded:
        if dataset == '0':
            vocab, priors, Q, H, S, A, C = _recalculate(num_topics, vocab_size)
            pickle.dump((num_topics, vocab_size, vocab, priors, Q, H, S, A, C), open('saved/tri.p', 'wb'))
        elif dataset == '1':
            vocab, priors, Q, H, S, A, C = _recalculate2(num_topics, vocab_size)
            pickle.dump((num_topics, vocab_size, vocab, priors, Q, H, S, A, C), open('saved/NIPS.p', 'wb'))


    # Print out some helpful information about the discovered topics
    print("Anchor Words:")
    print(list(map(lambda x: vocab[x], S)))
    print()
    print("Key: words are given as '<word> - (<P(word | topic)>, <P(topic | word)>, <prior P(word)>)'")

    topic_priors = numpy.zeros(A.shape[1])
    for i in range(len(topic_priors)):
        for word in range(A.shape[0]):
            # Solve by marginalizing over words and applying product rule
            topic_priors[i] += priors[word] * C[word, i]

    tops = top_words(A, num_words=20)
    coherences = calculate_coherences(tops, H)
    order = map(lambda x: x[0], sorted(zip(range(len(tops)), coherences), key=lambda x: x[1]))

    for i in order:
        print("Topic " + str(i+1) + ": anchor='%s', P(topic)=%.5f, coherence=%.2f" % (vocab[S[i]], topic_priors[i], coherences[i]))
        print(list(map(lambda x: vocab[x[0]] + " - (%.5f, %.5f, %.5f)" % (x[1], C[x[0], i], priors[x[0]]), tops[i])))
        print()

    print("Anchor word distances:")
    for anchor1 in S:
        dists = [numpy.linalg.norm(Q[anchor1] - Q[anchor2]) for anchor2 in S]
        print(', '.join([('%.3f' % dist) for dist in dists]))
    print()

