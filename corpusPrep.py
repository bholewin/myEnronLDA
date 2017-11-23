#References: https://radimrehurek.com/gensim/wiki.html#latent-dirichlet-allocation
#https://www.cs.rpi.edu/~szymansk/theses/ozcaglar.08.ms.pdf
#https://github.com/coreylynch/EnronTopicModelling/blob/master/enroncorpus.py

import logging, sys, os, string
from os import chdir
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import TfidfModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger('gensim.corpora.wikicorpus')
#*********************************************************#
#Text parsing functions
#*********************************************************#
def word_separator(documents):
    docs = word_tokenize(documents)
    return docs

def lowercase(documents):
    return [x.lower() for x in documents]

def remove_punctuation(documents):
    document = ["".join(c for c in s if c not in string.punctuation) for s in documents]
    return document

def remove_filler(documents):
    stop_words = stopwords.words('english')
    return [word for word in documents if word not in stop_words]

def remove_numbers(documents):
    no_ints = [x for x in documents if not x.isdigit()]
    return no_ints

def remove_empty(documents):
    no_blank = [x for x in documents if x]
    return no_blank
#*********************************************************#
#Corpus Class
#*********************************************************#
class EnronCorpus(TextCorpus):
    def __init__(self, dictionary=None, no_below=20):
        keep_words = 25000
        self.metadata = None
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
            self.dictionary.filter_extremes(no_below=no_below, no_above=0.1, keep_n=keep_words)
        else:
            self.dictionary = dictionary

    def get_texts(self, return_raw=False):
        length = 0
        chdir("/home/bryce/Corpus/data/")
        cwd = os.getcwd()
        for file in os.listdir(cwd):
            length += 1
            with open(file) as f:
                content = f.read()
                tokens = word_separator(content)
                l_docs = lowercase(tokens)
                better_tokens = remove_punctuation(l_docs)
                filler_free = remove_filler(better_tokens)
                num_free = remove_numbers(filler_free)
                pure = remove_empty(num_free)
                yield pure
        self.length = length


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)

    enron = EnronCorpus()
    output_file = raw_input("Enter a filename prefix: ")
    chdir("/home/bryce/Corpus/results")
    #save dict and bag of words
    #already saved
    enron.dictionary.save_as_text(output_file+'_ids.txt')
    MmCorpus.serialize(output_file+'_ids_bow.mm', enron, progress_cnt=10000)


    #intialize corpus reader and word-id mapping
    id2token = Dictionary.load_from_text(output_file+'_ids.txt')
    mm = MmCorpus(output_file+'_ids_bow.mm')

    #Build the tfidf
    tfidf = TfidfModel(mm, id2word=id2token, normalize=True)

    #save tfidf vectors
    MmCorpus.serialize(output_file+'_ids_tfidf.mm', tfidf[mm])

    del enron
    logger.info("Program complete.")

