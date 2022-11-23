"""
Supporting stand-alone functions

Authors: Alexander Katrompas, Vangelis Metsis
Organization: Texas State University

"""

import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import cfg

def get_simple_args():
    """
    Get command line arguments at program start.
    Return: the set flags: model, verbose, graph

    Usage: main.py [-vg]
           (assuming python3 in /usr/bin/)
    v: verbose mode (optional)
    g: graphing mode (optional)

    """
    
    # set optional defaults in case of error or no parameters
    verbose = cfg.VERBOSE
    graph = cfg.GRAPH
    
    if len(sys.argv) == 2 and re.search("^-[vg]+", sys.argv[1]):
        if 'v' in sys.argv[1]:
            verbose = True
        if 'g' in sys.argv[1]:
            graph = True

    return verbose, graph


def load_data(train_name, test_name, valid_name="", labels = 1, norm = False, index = None, header = None):
    """
    Load a dataset from csv. File is assumed to be in the form
    timesteps (rows) X features + labels (columns). All features are assumed to
    be before all labels (i.e. labels are the last columns)
    
    @param (string) train_name : training file name
    @param (string) test_name : test file name
    @param (string) valid_name : validation file name (optional)
    @param (int) labels : number of features
    @param (bool) norm : normalize the data
    @param (int) index : presence of an index (none or column number)
    @param (int) header : presence of aheader (none or row number)

    Return: datasets as numpy arrays
    """
    
    train = pd.read_csv(train_name, index_col=index, header=header).astype(float, errors='ignore')
    test = pd.read_csv(test_name, index_col=index, header=header).astype(float, errors='ignore')
    if valid_name:
        valid = pd.read_csv(valid_name, index_col=index, header=header).astype(float, errors='ignore')
    
    # if number of features not defined, assume columns -1
    features = train.shape[1] - labels

    train_X = train.iloc[:, 0:features]
    train_Y = train.iloc[:,features:]
    del train

    test_X = test.iloc[:, 0:features]
    test_Y = test.iloc[:,features:]
    del test

    if valid_name:
        valid_X = valid.iloc[:, 0:features]
        valid_Y = valid.iloc[:,features:]
        del valid
    else:
        valid_X = test_X.copy()
        valid_Y = test_Y.copy()

    if norm:
        [train_X[col].update((train_X[col] - train_X[col].min()) / (train_X[col].max() - train_X[col].min())) for col in train_X.columns]
        [test_X[col].update((test_X[col] - test_X[col].min()) / (test_X[col].max() - test_X[col].min())) for col in test_X.columns]
        [valid_X[col].update((valid_X[col] - valid_X[col].min()) / (valid_X[col].max() - valid_X[col].min())) for col in valid_X.columns]

    return train_X.to_numpy(), train_Y.to_numpy(), test_X.to_numpy(), test_Y.to_numpy(), valid_X.to_numpy(),  valid_Y.to_numpy()


def print_stats(filename, label_count=2):
    if label_count == 1:
        label_count = 2

    lines = 0
    fin = open(filename, "r")
    while fin.readline():
        lines += 1
    fin.close()

    y_true = np.zeros([lines], dtype=int)
    y_pred = np.zeros([lines], dtype=int)

    fin = open(filename, "r")
    i = 0
    for line in fin:
        line = line[:-1].split(",")
        y_true[i] = int(float(line[0]))
        y_pred[i] = int(float(line[1]))
        i += 1
    fin.close()

    target_names = []
    for i in range(label_count):
        target_names.append(str(i))

    print(classification_report(y_true, y_pred, target_names=target_names))

    return
