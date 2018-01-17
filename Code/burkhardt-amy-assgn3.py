from collections import Counter
import re
import math
import sys


def get_training(path):
    dat = []
    with open(path) as f:
        for line in f:
            line = re.split('\s', line)[1:-1] #ignore first and last element in the list.
            line = [x.upper() for x in line]
            line = [re.sub(r'[^A-Za-z|^0-9]+', '',x) for x in line]
            line = [word for word in line if word != '']
            dat.append(line)

    new_list = [item for sublist in dat for item in sublist]
    return new_list


class NaiveBayes:
    """ Implement Naive Bayes Estimator.
    """

    def __init__ (self, cat1, cat2):
        """Arguments: List of cat1 (POS) training data and list of cat2 (NEG) training data """

        self.merge = cat1 + cat2
        self.V =  len(set(self.merge))
        self.prior = math.log2(.5)

        c1_count = dict(Counter(cat1))
        c1_total = len(cat1)
        c2_count = dict(Counter(cat2))
        c2_total = len(cat2)
        c1_conditional = self.conditional_prob(c1_count, c1_total)
        c2_conditional = self.conditional_prob(c2_count, c2_total)
        self.conditionals = [c1_conditional, c2_conditional]

    def conditional_prob (self, count, total):
        keys = []
        prob = []
        for word in set(self.merge):
            try:
                probs = math.log2((count[word] + 1) / (total + self.V))
            except:
                probs = math.log2((1) / (total + self.V))

            keys.append(word)
            prob.append(probs)

        conditional = dict(zip(keys, prob))
        return conditional

    def compute_prob_test (self, test):
        prob_list = []
        for cat in self.conditionals:
            prob = self.prior
            for word in test:
                if word in cat:
                    prob = prob + cat[word]
            prob_list.append(prob)
        if prob_list[0] > prob_list[1]:
            result = 'POS'
        else:
            result = 'NEG'
        return result

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Missing arguments.")
        print("Usage: burkhardt-amy-assgn3.py train_POS_file.txt train_NEG_file.txt testfile.txt > burkhardt-amy-assgn3-test-output.txt")

    else:
        cat1_filename = sys.argv[1]
        cat2_filename = sys.argv[2]
        testing_filename = sys.argv[3]

        cat1 = get_training(cat1_filename)
        cat2 = get_training(cat2_filename)

        nb = NaiveBayes(cat1, cat2)

        with open(testing_filename) as f:
            for line in f:
                review = re.split(' |\t|\n', line)
                review_id = review[0]
                review_text = review[1:]
                print(review_text)
                review_text = [x.upper() for x in review_text]
                review_text = [re.sub(r'[^A-Za-z|^0-9]+', '', x) for x in review_text]
                review_text = [word for word in review_text if word != '']
                print(review_text)
                result = nb.compute_prob_test(review_text)
                print("{}\t{}".format(review_id, result))



