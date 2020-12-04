import matplotlib.pyplot as plt
import math
import os
import time
import operator
from collections import defaultdict

POS_LABEL = 'pos'
NEG_LABEL = 'neg'


def tokenize_doc(doc):
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)



def n_word_types(word_counts):
    return len(word_counts)


def n_word_tokens(word_counts):
    count=0
    for i in word_counts:
        count+=word_counts[i]
    return int(count)



class NaiveBayes:
    def __init__(self, path_to_data, tokenizer):
        self.vocab = set()
        self.path_to_data = path_to_data
        self.tokenize_doc = tokenizer
        self.train_dir = os.path.join(path_to_data, "train")
        self.test_dir = os.path.join(path_to_data, "test")
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self):
        pos_path = os.path.join(self.train_dir, POS_LABEL)
        neg_path = os.path.join(self.train_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        print ("REPORTING CORPUS STATISTICS")
        print ("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print ("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        self.class_total_doc_counts[label]+=1
        for i in bow:
            if i in self.class_word_counts[label]:
                self.class_word_counts[label][i]+=bow[i]
            else:
                self.class_word_counts[label][i]=bow[i]
            self.class_total_word_counts[label]+=bow[i]
            if i not in self.vocab:
                self.vocab.add(i)  
        

    def tokenize_and_update_model(self, doc, label):
        bow=tokenize_doc(doc.lower())
        self.update_model(bow,label)

    def top_n(self, label, n):
        li=[(k, v) for k, v in self.class_word_counts[label].items()]
        li.sort(key=lambda x:x[1],reverse=True)
        return li[0:n-1]

    def p_word_given_label(self, word, label):
        total=0
        for i in self.class_word_counts[label]:
            total+=self.class_word_counts[label][i]
        return self.class_word_counts[label][word]/total

    def p_word_given_label_and_alpha(self, word, label, alpha):
        total=0
        p=len(self.class_word_counts[label])
        return (self.class_word_counts[label][word]+alpha)/(self.class_total_word_counts[label]+(p*alpha))

    def log_likelihood(self, bow, label, alpha):
        total=0
        for i in bow:
            total+=math.log(self.p_word_given_label_and_alpha(i,label,alpha))
        return total
        
        

    def log_prior(self, label):
        total = self.class_total_doc_counts[POS_LABEL]+self.class_total_doc_counts[NEG_LABEL]
        return math.log(self.class_total_doc_counts[label]/total)

    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_prior(label) + self.log_likelihood(bow,label,alpha)

    def classify(self, bow, alpha):
        p=self.unnormalized_log_posterior(bow,POS_LABEL,alpha)
        n=self.unnormalized_log_posterior(bow,NEG_LABEL,alpha)
        if p>n:
            return POS_LABEL
        else:
            return NEG_LABEL


    def likelihood_ratio(self, word, alpha):
        return self.p_word_given_label_and_alpha(word,POS_LABEL,alpha)/ self.p_word_given_label_and_alpha(word,NEG_LABEL,alpha)

    def evaluate_classifier_accuracy(self, alpha):
        correct = 0.0
        total = 0.0

        pos_path = os.path.join(self.test_dir, POS_LABEL)
        neg_path = os.path.join(self.test_dir, NEG_LABEL)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            for f in os.listdir(p):
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    bow = self.tokenize_doc(content)
                    if self.classify(bow, alpha) == label:
                        correct += 1.0
                    total += 1.0
        return 100 * correct / total
