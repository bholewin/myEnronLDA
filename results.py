#Reference:http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/Gensim%20Newsgroup.ipynb
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from corpusPrep import *
import gensim

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    enron = EnronCorpus()
    # Provides the visuals of our LDA model
    chdir("/home/bryce/Corpus/results")
    #we have a 10 and 100 topic results processed
    num_topics = raw_input("Select the number of topics file you want to load: ")
    output_file = raw_input("Select the file prefix to pull from: ")
    lda = gensim.models.LdaModel.load('enron_lda'+str(num_topics)+'.pk1')
    id2word = gensim.corpora.Dictionary.load_from_text(output_file + '_ids.txt')
    vis_data = gensimvis.prepare(lda, enron, id2word)
    pyLDAvis.display(vis_data)

    #https://radimrehurek.com/topic_modeling_tutorial/3%20-%20Indexing%20and%20Retrieval.html