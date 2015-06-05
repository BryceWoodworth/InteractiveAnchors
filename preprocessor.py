import numpy
import os
import linecache
import math

__author__ = 'Bryce Woodworth'

def _mapper(word):
    """Takes a word and maps it to the lowercase equivalent
    with punctuation removed."""
    punctuation = ',.!?~#()[]-+='
    return word.strip(punctuation).lower()


def stop_remover(filenames, stopwords, outdir):
    """Removes the given stop words from the given files,
    as well as punctuation and capitalization,
    and saves the result in the given directory.

    Inputs:
        filenames: A list of paths to files to be cleaned
        stopwords: A list of words to be removed
        outdir: The directory to save the cleaned files in

    Returns: nothing
    """
    for filename in filenames:
        cleaned = ""
        with open(filename) as file:
            try:
                for line in file:
                    line = map(_mapper, line.split())
                    line = filter(lambda w: w not in stopwords, line)
                    cleaned += (' '.join(line) + '\n')
            except UnicodeDecodeError:
                print(filename)
                continue

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(os.path.join(outdir, os.path.basename(filename)), 'w') as outfile:
            outfile.write(cleaned)


def find_most_common(filenames, num_words, ignore=[]):
    """Finds the n most common words in the given files
    and returns them as a list, ignoring words in the
    ignore list.

    Inputs:
        filenames: A list of paths to files to be searched through
        num_words: The number of most common words to find
        ignore: A list of words that should not be considered

    Returns: A list of the num_words most common words in filenames
    """
    freqs = {}
    for filename in filenames:
        with open(filename) as file:
            try:
                for line in file:
                    for word in line.split():
                        if word in ignore:
                            continue
                        if word in freqs:
                            freqs[word] += 1
                        else:
                            freqs[word] = 1
            except UnicodeDecodeError:
                print(filename)
                continue

    sort = sorted(freqs.items(), key=lambda x: x[1])[-num_words:]
    sort.reverse()
    return [word for (word, _) in sort]


def word_filter(filenames, vocab, outdir):
    """Removes all words that aren't in the vocab
    from filenames and places the modified files in outdir.

    Inputs:
        filenames: A list of paths to files to be pruned
        vocab: The vocabulary to prune the files to
        outdir: The path to place the modified files in, using their original names

    Returns: void
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename in filenames:
        pruned = ""
        with open(filename) as file:
            try:
                for line in file:
                    line = ' '.join(filter(lambda w: w in vocab, line.split()))
                    pruned += line + '\n'
            except UnicodeDecodeError:
                continue

        with open(os.path.join(outdir, os.path.basename(filename)), 'w') as outfile:
            outfile.write(pruned)


def word_bag_reduce(filename, vocname, dictname, size, stopwords=None):
    """Treats the input file as a bag of words (formatting specific)
    and outputs its co-occurrence matrix Q, using a vocab
    size of size by removing the least common words overall, and
    optionally removing stopwords from the specified file.

    Inputs:
        filename: A path to the bag of words file
        vocname: A path to the indexed vocabulary file
        dictname: A path to a dictionary to filter words by
        size: The number of words to keep
        stopwords: The path to a file containing whitespace-separated stop words,
                   or None if no stop words

    Returns:
        Q: The word-word co-occurrence matrix
        vocab: The list of words indexed by their Q index
    """
    # First get the indices of the stop words, if any
    stops = []
    stop_indices = []
    stop_set = set()
    if stopwords is not None:
        with open(stopwords) as stopfile:
            for line in stopfile:
                stops.extend(line.split())

        # Convert from string stopwords to dictionary index stopwords
        with open(vocname) as vocfile:
            lineindex = 1
            for line in vocfile:
                if line.strip() in stops:
                    stop_indices.append(lineindex)
                lineindex += 1
        stop_set = set(stop_indices)

    # Get the total word counts into a dictionary
    counts = dict()
    with open(filename) as file:
        num_docs = int(file.readline())
        cur_vocab = int(file.readline())
        file.readline()

        for line in file:
            split = line.split()
            # only consider words that aren't stopwords
            if split[1] not in stops:
                try:
                    counts[split[1]] += int(split[2])
                except KeyError:
                    counts[split[1]] = int(split[2])

    # Next get all of the dictionary words from dictname
    dictionary = []
    with open(dictname) as dictfile:
        for line in dictfile:
            dictionary.append(line.strip())

    # Now use that dictionary to find the most common words
    sort = list(filter(lambda w: linecache.getline(vocname, int(w)).strip() in dictionary, sorted(counts, key=lambda x: counts[x], reverse=True)))

    # Get the mapping from sorted index to string
    vocab = list(map(lambda x: linecache.getline(vocname, int(x)).strip(), sort[:size]))

    with open(filename) as file:
        file.readline()
        file.readline()
        file.readline()

        H = numpy.zeros((size, num_docs))  # The term-by-document matrix
        for line in file:
            split = line.split()
            try:
                ind = sort.index(split[1])
            except ValueError:
                continue
            if ind < size:
                H[ind][int(split[0])-1] += int(split[2])

    # recover the word-word co-occurrence matrix Q
    Q = construct_Q(H)

    priors = [0.0] * Q.shape[0]

    for i in range(len(priors)):
        priors[i] = numpy.sum(Q[i])
    priors = priors / sum(priors)

    Qbar = numpy.empty_like(Q)
    for row in range(Q.shape[0]):
        total = numpy.sum(Q[row])
        if total <= 0.0:
            print("invalid sum of %d at row %d" % (total, row))
        Qbar[row] = Q[row] / total

    return vocab, priors, Qbar, H


def construct_Q(H):
    """
    Constructs the Q matrix from the word-document counts H

    :param H: the word-document counts, size V x K
    :return: Q: The estimated word-word co-occurence matrix
    """
    Hbar = numpy.empty_like(H)
    Hhat = numpy.zeros((H.shape[0], H.shape[0]))
    n = numpy.sum(H, axis=0)

    for col in range(H.shape[1]):
        denom = n[col] * (n[col] - 1)
        if denom <= 0.0:
            print("Invalid denominator of %d at column %d. Ignoring." % (denom, col))
        else:
            Hbar[:, col] = H[:, col] / math.sqrt(denom)
            Hhat += (numpy.diag(H[:, col]) / denom)

    # recover the word-word co-occurrence matrix Q
    return numpy.dot(Hbar, Hbar.transpose()) - Hhat



if __name__ == '__main__':
    base_dir = '/home/ephax/code/datasets/SentenceCorpus/unlabeled_articles'
    stop_file = '/home/ephax/code/datasets/SentenceCorpus/word_lists/stopwords.txt'
    clean_dir = '/home/ephax/code/datasets/SentenceCorpus/cleaned'
    prune_dir= '/home/ephax/code/datasets/SentenceCorpus/pruned'
    vocab_size = 1000

    with open(stop_file) as stopfile:
        stops = [stop.strip() for stop in stopfile]

    dirs = [os.path.join(base_dir, dir) for dir in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, dir)) and not dir.startswith('.')]

    allfiles = [os.path.join(direct, file) for direct in dirs for file in os.listdir(direct)
                if os.path.isfile(os.path.join(direct, file)) and not file.startswith('.')]

    for dir in dirs:
        files = [os.path.join(dir, file) for file in os.listdir(dir)
                 if os.path.isfile(os.path.join(dir, file)) and not file.startswith('.')]
        cleandir = os.path.join(clean_dir, os.path.basename(dir))
        prunedir = os.path.join(prune_dir, os.path.basename(dir))
        stop_remover(files, stops, cleandir)
        files = [os.path.join(cleandir, file) for file in os.listdir(cleandir)
                 if os.path.isfile(os.path.join(cleandir, file)) and not file.startswith('.')]
        print("Files have been stripped of stop words")

        voc = find_most_common(allfiles, vocab_size)
        word_filter(files, voc, prunedir)
        print("Files have been stripped of words not in vocab\n")




