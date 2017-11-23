#Reference: https://radimrehurek.com/gensim/wiki.html#latent-dirichlet-allocation
#running the LDA model

import logging, gensim
from os import chdir


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    chdir("/home/bryce/Corpus/results")
    output_file = raw_input("Enter a filename prefix: ")

    #load id-word mapping (dict)
    id2word = gensim.corpora.Dictionary.load_from_text(output_file+'_ids.txt')

    #load corpus iterator
    mm = gensim.corpora.MmCorpus(output_file+'_ids_tfidf.mm')
    print(mm)

    #9 topics are chosen per the results from https://www.cs.rpi.edu/~szymansk/theses/ozcaglar.08.ms.pdf
    num_topics = raw_input("Enter the number of topics for the model: ")
    type = 'batch'
    passes = 20
    update_every = 0
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=num_topics, update_every=update_every,
                                          passes=passes)

    lda.save('enron_lda'+str(num_topics)+'.pk1')
    lda.print_topics(num_topics)

