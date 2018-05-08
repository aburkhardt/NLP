#!/usr/bin/env python
"""
This module provides some helper functions for training classifiers in the associated jupyter notebooks.
"""

import re
import string
import numpy as np


def simple_tokenizer(string):
    """
    Simply tokenize on whitespace.
    """
    return string.split()

def normalize_tweet(item):
    """
    This function converts a document to lower case, identifies hashtags and
    URLs, and removes punctuation.

    Some ideas taken from (TODO more should be implemented):
    https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    Arguments:
        item: (str) the text (a sample document) to normalize

    Returns:
        (str) the normalized text
    """

    item = item.lower()

    # remove quotation marks punctuation
    item = re.sub(r'["]', '', item)
    
    # Replace URLs with [URL] tag
    item = re.sub(r'http://\S+', r'_URL_', item)

    # replace hashtag with [HASH] and tag without # sign
    item = re.sub(r'#(\S+)(\s?)', r'_HASH_ \1\2', item)

    # replace @user with '[USER] user'
    item = re.sub(r'@(\S+)', r'_USER_ \1', item)

    # replace numbers with [NUMBER]
    item = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '_NUMBER_', item)

    return item

def normalize_simple(item):
    """
    This function converts a document to lower case and strips out all punctuation.

    Arguments:
        item: (str) the text (a sample document) to normalize

    Returns:
        (str) the normalized text
    """
    item = item.lower()
     # remove punctuation
    item = re.sub("[%s]" % string.punctuation, '', item)

    return item

def parse_training_data(paths, labels=['NEG', 'POS']):
    """
    Get training data into a list of documents and a corresponding list of target labels.

    Note that for binary classification, scikit-learn's score functions assume
    by default that the negative label is at index 0 and the positive label is
    at index 1.

    Use like this:

    training_docs, training_labels = parse_training_data(['path/to/NEG.txt', 'path/to/POS.txt'])


    Arguments:
        paths: [str] a list of file names to read
        labels: [str] a list of labels corresponding to the file paths

    Returns:
        ([str], [str]) a 2-tuple containing the list of sample documents
        extracted from the given files and their corresponding labels.
    """
    if type(paths) != list:
        paths = [paths]
    if type(labels) != list:
        labels = [labels]

    if len(paths) != len(labels):
        print("paths and labels must be the same length.")
        return ([], [], -1)
    
    train_data = []
    train_target = []

    for i, path in enumerate(paths):
        docs = parse_data_file(path)
        docs = [doc for ident, doc in docs]
        train_data.extend(docs)
        train_target.extend([labels[i]] * len(docs))

    return (train_data, train_target)

def parse_data_file(path):
    """
    Read document samples from file at path, returning a list of (id, text)
    2-tuples for each document.

    One document per line, with the id and text fields separated by a tab.

    Discards any malformed rows in input files. This takes care of any header
    line (where the ID field is non-numeric).
    """
    documents = []
    with open(path) as f:
        for line in f:
            fields = line.split("\t")

            if len(fields) != 2:
                # this line ain't right; discard it
                #print(line)
                continue

            try:
                id_field = int(fields[0])
            except ValueError:
                # first field is non-numeric (probably a header row); discard it
                #print(line)
                continue

            documents.append((id_field, fields[1]))

    return documents

def evaluate_file(path, estimator):
    """
    Run an sklearn estimator on the documents contained in the file at path.

    Returns a list of documents for each class (that is a list of lists, where
    the first list corresponds to class 0, etc.
    """
    documents = parse_data_file(path)
    docs = [doc for id, doc in documents]
    predictions = estimator.predict(docs)

    results = {}
    for i, pred in enumerate(predictions):
        # test if dict key for this class exists yet
        try:
            results[pred]
        except KeyError:
            results[pred] = []

        results[pred].append(documents[i])

    return results


