


# imports file Naive Bayes
from Naive_Bayes import *

# download the IMDB large movie review corpus from github to a file location on your computer

PATH_TO_DATA = '/Users/parthpatel/Downloads/large_movie_review_dataset'  # set this variable to point to the location of the IMDB corpus on your computer
POS_LABEL = 'pos'
NEG_LABEL = 'neg'
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


#Train the model
nb = NaiveBayes(PATH_TO_DATA, tokenizer=tokenize_doc)
nb.train_model()


#Calculate the probability of words in positive review and negative review
print("P('amazing'|pos):",  nb.p_word_given_label("amazing", POS_LABEL))
print("P('amazing'|neg):",  nb.p_word_given_label("amazing", NEG_LABEL))
print("P('dull'|pos):",  nb.p_word_given_label("dull", POS_LABEL))
print("P('dull'|neg):",  nb.p_word_given_label("dull", NEG_LABEL))



# Calculate likelihood ratio of words
print ("LIKELIHOOD RATIO OF 'amazing':", nb.likelihood_ratio('amazing', 0.2))
print ("LIKELIHOOD RATIO OF 'dull':", nb.likelihood_ratio('dull', 0.2))
print ("LIKELIHOOD RATIO OF 'and':", nb.likelihood_ratio('and', 0.2))
print ("LIKELIHOOD RATIO OF 'to':", nb.likelihood_ratio('to', 0.2))




#CLassify reviews by providing smoothing paramter 0.2 (bias) to every word
print(nb.classify_reviews(0.2))







