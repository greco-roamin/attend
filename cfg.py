"""
Run Time Configuration

Authors: Alexander Katrompas
Organization: Texas State University

"""

# ###############################
# defines
# ###############################
DATAPATH = 'data/'
DATA = 2

#########################
# Two simple data sets are provided for simple validation of
# code. Complete data set sources are found cited in the paper.
#########################

if DATA == 1:
    # For use with seq-to-vec
    TRAIN = DATAPATH + "weatherAUStrain_tg.csv"
    TEST = DATAPATH + "weatherAUStest_tg.csv"
    VALID = ""  # optional, if none given test set will be used for validation
    OUTPUT = 1
    INDEX = 0
    HEADER = 0
    SEQLENGTH = 5
    STRIDE = 1  # 1 is min stride, max overlap. SEQLENGTH is max stride, min overlap
    FIXED = 0
elif DATA == 2:
    TRAIN = DATAPATH + "aq-train_tg.csv"
    TEST = DATAPATH + "aq-test_tg.csv"
    VALID = ""
    OUTPUT = 1
    INDEX = 0
    HEADER = 0
    SEQLENGTH = 20
    STRIDE = 1 # 1 is min stride, max overlap. SEQLENGTH is max stride, min overlap
    FIXED = 0

# ###############################
# command line parameter defaults
# ###############################
VERBOSE = False
GRAPH = False

# ###############################
# Hyperparameters
# ###############################

# General
BATCH_SIZE = 32  # not used, TF default is used
SHUFFLE = True
BINTHRESHOLD = 0.5
EAGERLY = True # must be true for attention capture
BIDIRECTIONAL = True
LSTM = 64
DENSE1 = 128
DENSE2 = 64
DENSE3 = 32
DENSE4 = 32
DROPOUT = .25

EPOCHS = 2
PATIENCE = 2

# these are mutually exclusive [0,1], set only one to 1
NO_ATTN = 0
SIMPLE_ATTN = 1
SELF_ATTN = 0

# weak sanity check for the above constants
if NO_ATTN + SIMPLE_ATTN + SELF_ATTN != 1:
    raise Exception("Attention not specified correctly")

PREDICT = True

# #######################
# Attention capture files
# #######################
# all attention layer activations
VATTEND = "vattend.csv"

# normalization of vattend.csv on a per seq basis
VATTEND_NORM = "vattend_norm.csv"

# to capture predictions for prediction validation
VOUT= "valid_out.csv"

# to capture predictions at the time of attention capture
VATTNOUT= "valid_attn_out.csv"