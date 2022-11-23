#!/usr/bin/python3

"""
Author: Alexander Katrompas
Organization: Texas State University

Usage: main2.py [-vg]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)
g: graphing mode (optional)

seq-to-vec modeling
"""

# python libs
from pathlib import Path
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot
from numpy import argmax
import numpy as np
import re

# local libs
import cfg
import dml
import functions as fn
from trainCb import TrainCb
from predictCb import PredictCb
from attnSelf import AttnSelf # self attention from AIME paper
from attention import Attention # simple attention
from attentionSelf import AttentionSelf # self attention from # https://pypi.org/project/keras-self-attention/0.0.10/

np.set_printoptions(precision=2, suppress=True)

# get command line arguments
verbose, graph = fn.get_simple_args()

print("TF Version:", tf. __version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print()
print("Executing Seq-to-Vec LSTM ", end="")
if cfg.NO_ATTN:
    print("without Attention...")
elif cfg.SIMPLE_ATTN:
    print("with Simple Attention...")
elif cfg.SELF_ATTN1:
    print("with Self Attention 1...")
elif cfg.SELF_ATTN2:
    print("with Self Attention 2...")

if verbose: print(" - Verbose: On")
else: print(" - Verbose: Off")
if graph: print(" - Graphing: On")
else: print(" - Graphing: Off")

#############################
# Load and Shape Data
#############################

if not cfg.FIXED:
    # for data that is continuous and in order

    train_X, train_Y, test_X, test_Y, valid_X, valid_Y_temp = fn.load_data(cfg.TRAIN, cfg.TEST, cfg.VALID, norm=True,
                                                                           labels=cfg.OUTPUT, index=cfg.INDEX,
                                                                           header=cfg.HEADER)
    if cfg.OUTPUT == 1:
        valid_Y = valid_Y_temp.flatten() # needs to be flat for prediction
    else:
        valid_Y_len = valid_Y_temp.shape[0] # get number of rows
        valid_Y = np.zeros([valid_Y_len]) # make array to hold one label per row
        for i in range(valid_Y_len): # for each row
            valid_Y[i] = argmax(valid_Y_temp[i]) # get argmax for categorical label
    del valid_Y_temp

    if verbose:
        print("files loaded...")
        print("  Train:", Path(cfg.TRAIN).name)
        print("  Test: ", Path(cfg.TEST).name)
        if(cfg.VALID):
            print("  Valid:", Path(cfg.VALID).name)
        else:
            print("  Valid: using test set")
        print()
        print("Shapes loaded...")
        print("  train X", train_X.shape)
        print("  train Y", train_Y.shape)
        print("  test_X", test_X.shape)
        print("  test_Y", test_Y.shape)
        print("  valid_X", valid_X.shape)
        print("  valid_Y", valid_Y.shape)
        print()

    # shift Y for time generator
    # because it uses the next Y after the sequence
    if train_Y.shape[1] > 1:
        train_Y = np.roll(train_Y, 1, axis=0)
        test_Y = np.roll(test_Y, 1, axis=0)
    else:
        train_Y = np.roll(train_Y, 1)
        test_Y = np.roll(test_Y, 1)
    # dupicate the second to the first because the last item is unrealated in time
    train_Y[0] = train_Y[1]
    test_Y[0] = test_Y[1]

    # don't shift validation since it's not used until prediction
    # and then it is matched up with sequence size.
    t_generator = TimeseriesGenerator(train_X, train_Y, length=cfg.SEQLENGTH, stride=cfg.STRIDE, batch_size=cfg.BATCH_SIZE)
    v_generator = TimeseriesGenerator(test_X, test_Y, length=cfg.SEQLENGTH, stride=cfg.STRIDE, batch_size=cfg.BATCH_SIZE)

    if verbose:
        print('Generator train samples: %d' % len(t_generator))
        print('Generator test samples: %d' % len(v_generator))
        print('Sequence: %d' % cfg.SEQLENGTH)
        print('Stride: %d' % cfg.STRIDE)
        print()

else:
    # for data that is segmented and not necessarily in order
    train_X, train_Y, test_X, test_Y, valid_X, valid_Y = fn.load_seq_data(cfg.TRAIN, cfg.TEST, cfg.TEST, cfg.SEQLENGTH,
                                                                           labels=cfg.OUTPUT, index=cfg.INDEX,
                                                                           header=cfg.HEADER)

#############################
# Build Model
#############################

if not cfg.FUNCTIONAL:
    model = tf.keras.models.Sequential()

    if cfg.BIDIRECTIONAL:
        model.add(layers.Bidirectional(layers.LSTM(cfg.LSTM, input_shape=(cfg.SEQLENGTH,
                train_X.shape[1]),
                return_sequences=True,
                dropout=cfg.DROPOUT)))
    else:
        model.add(layers.LSTM(cfg.LSTM,
                return_sequences=True,
                dropout=cfg.DROPOUT))

    if cfg.NO_ATTN:
        model.add(layers.TimeDistributed(layers.Dense(cfg.DENSE1, activation='sigmoid')))
    elif cfg.SIMPLE_ATTN:
        model.add(Attention(return_sequences= not cfg.KSUM))
    elif cfg.SELF_ATTN1:
        model.add(AttnSelf(cfg.SEQLENGTH, return_sequences= not cfg.KSUM))
    elif cfg.SELF_ATTN2:
        model.add(AttentionSelf(units = cfg.LSTM, attention_width = cfg.SEQLENGTH))

    if cfg.FLATTEN:
        model.add(layers.Flatten())

    model.add(layers.Dense(cfg.DENSE1, activation='sigmoid'))
    if cfg.DROPOUT:
        model.add(layers.Dropout(cfg.DROPOUT))
    model.add(layers.Dense(cfg.DENSE2, activation='sigmoid'))
    if cfg.DROPOUT:
        model.add(layers.Dropout(cfg.DROPOUT))
    model.add(layers.Dense(cfg.OUTPUT, activation='sigmoid' if cfg.OUTPUT == 1 else 'softmax'))

    #############################
    # Finalize Model and Train
    #############################
    if cfg.OUTPUT == 1:
        model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], run_eagerly=cfg.EAGERLY)
    else:
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=cfg.EAGERLY)

    if verbose:
        print("Epochs:", cfg.EPOCHS)

    trainCb = TrainCb()
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True, verbose=verbose)
else:
    #visible = layers.Input(shape=(cfg.SEQLENGTH, train_X.shape[1]))

    #lstm1 = layers.Bidirectional(layers.LSTM(cfg.LSTM, input_shape=(cfg.SEQLENGTH,
     #           train_X.shape[1]),
     #           return_sequences=True,
     #           dropout=cfg.DROPOUT))
    #sattn = Attention(return_sequences= not cfg.KSUM)(lstm1)

    #lstm2 = layers.Bidirectional(layers.LSTM(cfg.LSTM, input_shape=(cfg.SEQLENGTH,
     #           train_X.shape[1]),
     #           return_sequences=True,
     #           dropout=cfg.DROPOUT))
    #gattn = Attention(return_sequences=not cfg.KSUM)(lstm2)


    model1 = tf.keras.models.Sequential()
    model1.add(layers.Input(shape=(cfg.SEQLENGTH, 1)))
    model1.add(layers.Bidirectional(layers.LSTM(cfg.LSTM,
                                                   return_sequences=True,
                                                   dropout=cfg.DROPOUT)))
    model1.add(Attention(return_sequences=not cfg.KSUM))

    model2 = tf.keras.models.Sequential()
    model2.add(layers.Input(shape=(cfg.SEQLENGTH, 1)))
    model2.add(layers.Bidirectional(layers.LSTM(cfg.LSTM,
                                                   return_sequences=True,
                                                   dropout=cfg.DROPOUT)))
    model2.add(AttnSelf(cfg.SEQLENGTH, return_sequences=not cfg.KSUM))

    mergedOut = layers.Concatenate()([model1.output, model2.output])
    mergedOut = layers.Flatten()(mergedOut)
    mergedOut = layers.Dense(cfg.DENSE1, activation='sigmoid')(mergedOut)
    mergedOut = layers.Dropout(cfg.DROPOUT)(mergedOut)
    mergedOut = layers.Dense(cfg.DENSE2, activation='sigmoid')(mergedOut)
    mergedOut = layers.Dropout(cfg.DROPOUT)(mergedOut)

    # output layer
    mergedOut = layers.Dense(cfg.OUTPUT, activation='sigmoid' if cfg.OUTPUT == 1 else 'softmax')(mergedOut)

    model = Model([model1.input, model2.input], mergedOut)

    #############################
    # Finalize Model and Train
    #############################
    if cfg.OUTPUT == 1:
        model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], run_eagerly=cfg.EAGERLY)
    else:
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=cfg.EAGERLY)

    if verbose:
        print("Epochs:", cfg.EPOCHS)

    trainCb = TrainCb()
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, restore_best_weights=True, verbose=verbose)

if not cfg.FIXED:
    history = model.fit(t_generator,
                        # batch_size = cfg.BATCH_SIZE,
                        epochs=cfg.EPOCHS,
                        verbose=verbose,
                        shuffle=cfg.SHUFFLE,
                        validation_data=v_generator,
                        callbacks=[early_stopping, trainCb])
else:
    history = model.fit([train_X, train_X], train_Y,
                #batch_size = cfg.BATCH_SIZE,
                epochs=cfg.EPOCHS,
                verbose=verbose,
                shuffle=cfg.SHUFFLE,
                #validation_data=([train_X, train_X], test_Y),
                callbacks=[early_stopping, trainCb])

if verbose:
    model.summary()

#############################
# Graphing Loss
#############################
if graph and len(history.history['loss']) and len(history.history['val_loss']):
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

if not cfg.PREDICT: exit()

###################################
# Prediction and Validation
###################################
if verbose:
    print("Beginning Validation Prediction")
############################################
# make predictions based on model type
# outputs a 3D array in both cases, sequences x sequence x labels
############################################

if not cfg.FIXED:

    length = valid_X.shape[0]-cfg.SEQLENGTH+1
    predictions = []

    if verbose:
        tenpercent = int(length * .1)
        percent = -10

    for j in range(length):
        sequence = []
        for i in range(cfg.SEQLENGTH):
            sequence.append(valid_X[j+i])
        sequence = np.array(sequence)
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        yhat = model.predict(sequence)
        predictions.append(yhat)

        if verbose and not (float(j) % tenpercent):
            percent += 10
            print( percent, "% done...", sep="")

else:
    length = valid_X.shape[0]

    if verbose:
        tenpercent = int(length * .1)
        percent = -10

    predictions = []
    for i in range(length):
        sequence = valid_X[i].reshape(1, valid_X[i].shape[0], valid_X[i].shape[1])
        yhat = model.predict([sequence,sequence])
        predictions.append(yhat)

        if verbose and not (float(i) % tenpercent):
            percent += 10
            print( percent, "% done...", sep="")

predictions = np.array(predictions)

############################################
# get predictions based on number of outputs
############################################
if cfg.OUTPUT == 1:
    print("making single output predictions")
    # since a 3D array [[[]],[[]],[[]]...] just flatten to get 1D array
    predictions = predictions.flatten()
    # convert to binary
    for i in range(len(predictions)):
        if predictions[i] > cfg.BINTHRESHOLD:
            predictions[i] = 1
        else:
            predictions[i] = 0
else:
    print("making multiple output predictions")
    # since 3D array [[[],[]]...]] first turn into 2D array
    temp = predictions.reshape(predictions.shape[0]*predictions.shape[1], predictions.shape[2])

    # then turn into 1D array of argmax values
    predictions = []
    for value in temp:
        predictions.append(argmax(value))
    del temp
    predictions = np.array(predictions)

if verbose:
    print("Validation Prediction Complete")
    print("Made", len(predictions), "predictions")
    print()

############################################
# save predictions based on model
############################################
fout = open(cfg.VOUT, "w")

if not cfg.FIXED:
    for i in range(cfg.SEQLENGTH-1):
        # write out start for which we have no validation, dup actual
        fout.write(str(valid_Y[i]) + "," + str(valid_Y[i]) + "\n")

    for i in range(cfg.SEQLENGTH - 1, len(predictions)):
        # continue writing where last left off
        fout.write(str(valid_Y[i]) + "," + str(predictions[i - (cfg.SEQLENGTH - 1)]) + "\n")
else:
    for i in range(cfg.SEQLENGTH-1, len(predictions)):
        fout.write(str(argmax(valid_Y[i])) + "," + str(predictions[i-(cfg.SEQLENGTH-1)]) + "\n")

fout.close()

if verbose:
    print("Analysis Seq-to-Vec LSTM ", end="")
    if cfg.NO_ATTN:
        print("without Attention...")
    elif cfg.SIMPLE_ATTN:
        print("with Simple Attention...")
    elif cfg.SELF_ATTN1:
        print("with Self Attention 1...")
    elif cfg.SELF_ATTN2:
        print("with Self Attention 2...")
    print('Sequence: %d' % cfg.SEQLENGTH)
    print('Stride: %d' % cfg.STRIDE)

fn.print_stats(cfg.VOUT, cfg.OUTPUT)

#########################################
# Capture Attention Layer Representations
#########################################
if cfg.EAGERLY and cfg.SIMPLE_ATTN:

    if verbose:
        print("Beginning Attention Capture")

    # create all empty output files
    # for use in predictCb
    # bad design, but it works for now

    fout = open(cfg.VATTEND, "w")
    fout.close()

    fout = open(cfg.VATTEND_NORM, "w")
    fout.close() # normalization per seq from vattend_sum.csv


    vout = open(cfg.VATTNOUT,"w")
    if not cfg.FIXED:
        instances = valid_X.shape[0] - cfg.SEQLENGTH + 1
        print("  number of instances to write:", instances)
        for i in range(instances):
            seq = []
            for j in range(cfg.SEQLENGTH):
                seq.append(valid_X[i+j])
            seq = np.array(seq)
            seq = seq.reshape(1, seq.shape[0], seq.shape[1])
            predictCb = PredictCb()
            predictions = model.predict(seq, callbacks=[predictCb]).flatten()

            # convert predictions to 0,1
            for k in range(len(predictions)):
                if predictions[k] > cfg.BINTHRESHOLD:
                    predictions[k] = 1
                else:
                    predictions[k] = 0

            # write out to file
            vout.write(re.sub("\s+", ",", str(predictions)[1:-1])+"\n")
    else:
        instances = valid_X.shape[0]
        seq = valid_X[i].reshape(1, valid_X[i].shape[0], valid_X[i].shape[1])
        print("  number of instances to write:", instances)
        for i in range(instances):
            seq = valid_X[i].reshape(1, valid_X[i].shape[0], valid_X[i].shape[1])
            predictCb = PredictCb()
            predictions = model.predict([seq,seq], callbacks=[predictCb]).flatten()

            # convert predictions to 0,1
            for k in range(len(predictions)):
                if predictions[k] > cfg.BINTHRESHOLD:
                    predictions[k] = 1
                else:
                    predictions[k] = 0

            # write out to file
            vout.write(re.sub("\s+", ",", str(predictions)[1:-1])+"\n")


    vout.close()

    if verbose:
        print("Attention Capture Complete")
        print()
