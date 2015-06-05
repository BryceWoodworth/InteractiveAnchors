from random import randint, random
from time import time

__author__ = 'Bryce Woodworth'

def gibbs(documents, num_topics, num_simulations):
    """Performs Gibbs sampling from the given documents.
    It expects the documents to already be preprocessed,
    so there is no punctuation and words are lowercase.
    Stop words should also be removed.

    input:
        documents: an iterable of names of text files
        num_topics: the number of topics to model
        num_simulations: how many total simulations we want

    output: (z, doc_topic, topic_word, topics)
        z: the topic assignment for each word
        doc_topic: for each document, num words in topic
        topic_word: for each topic, num times word assigned
        topics: num times any word assigned to topic

    Note: This implementation requires a topic assignment
    to each word in the corpus and a list of all words,
    requiring more memory than the size of the corpus.
    Just FYI.
    """
    z = []              # store topic assignments here
    words = []          # the list of all words
    doc_topic = []      # for each document, num words in topic
    topic_word = []     # for each topic, num times word assigned
    topics = [0] * num_topics   # num times any word assigned to topic

    indices = []        # helper for finding which document a word's in

    for i in range(num_topics):
        topic_word.append({})   # initialize with empty dicts

    # Initialize random assignments and counts
    for document in documents:
        topic_counts = [0] * num_topics
        indices.append(len(words))  # this document starts this far in
        with open(document, 'r') as doc:
            for line in doc:
                for word in line.split():
                    assignment = randint(0, num_topics - 1)
                    z.append(assignment)
                    words.append(word)
                    topics[assignment] += 1
                    topic_counts[assignment] += 1
                    if word in topic_word[assignment]:
                        topic_word[assignment][word] += 1
                    else:
                        topic_word[assignment][word] = 1

        doc_topic.append(topic_counts)

    # Begin simulation
    for _ in range(num_simulations):
        _simulate(z, words, indices, doc_topic, topic_word, topics)

    return (z, doc_topic, topic_word, topics)


def _simulate(z, words, indices, doc_topic, topic_word, topics):
    doc = -1
    for i in range(len(z)):
        # keep track of which document we are in
        if doc + 1 < len(indices):
            if i >= indices[doc+1]:
                doc += 1

        # subtract out the value due to this variable
        word = words[i]
        topic = z[i]
        topics[topic] -= 1
        topic_word[topic][word] -= 1
        doc_topic[doc][topic] -= 1

        # find the distribution to draw the new topic assignment from
        weights = []
        total = 0
        for k in range(len(topics)):
            # Here I use symmetric dirichlet hyperparameters of 3
            # for simplicity and a somewhat dense distribution
            hp = 3
            # This is where the mathemagic happens
            math = hp
            if word in topic_word[k]:
                math += topic_word[k][word]
            weight = (doc_topic[doc][k] + hp) * math / (topics[k] + hp * len(words))
            total += weight
            weights.append(total)

        # Select a new topic from this distribution
        rand = random()*total
        new_topic = 0
        while rand > weights[new_topic]:
            new_topic += 1

        z[i] = new_topic
        topics[new_topic] += 1
        if word in topic_word[new_topic]:
            topic_word[new_topic][word] += 1
        else:
            topic_word[new_topic][word] = 1
        doc_topic[doc][new_topic] += 1


# Test it out!
if __name__ == "__main__":
    base = '/home/ephax/code/datasets/SentenceCorpus/unlabeled_articles/'
    doc_bases = ['arxiv_unlabeled/', 'jdm_unlabeled/']
    nums = [112, 2, 39, 4, 5, 62, 8, 9, 10, 100]
    morenums = nums + [1, 11, 111, 121, 113, 131, 114, 141, 115, 151]

    documents = [base + doc + str(num) + '.txt' for doc in doc_bases for num in nums]
    moredocuments = documents + [base + doc + str(num) + '.txt' for doc in doc_bases for num in morenums]
    for docs in [documents, moredocuments]:
        for num_topics in range(1, 20, 2):
            for iterations in range(1, 1000, 200):
                start_time = time()
                gibbs(docs, num_topics, iterations)
                total_time = time()-start_time
                print("%d\t%d\t%d\t%d" % (len(docs), num_topics, iterations, total_time))
