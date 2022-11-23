import tensorflow as tf
import os.path
import numpy as np
import cfg

class PredictCb(tf.keras.callbacks.Callback):
    def __init__(self, verbose = False):
        super().__init__()
        self.batch = 0
        self.verbose = verbose

        # files are created empty in main first
        # bad design, but it works for now
        if os.path.exists(cfg.VATTEND):
            self.fout1 = open(cfg.VATTEND, "a")
        else:
            raise Exception("file not found:", cfg.VATTEND)

        if os.path.exists(cfg.VATTEND_NORM):
            self.fout2 = open(cfg.VATTEND_NORM, "a")
        else:
            raise Exception("file not found:", cfg.VATTEND_NORM)

    def on_predict_begin(self, logs=None):
        pass

    # will accept and write one seq at a time
    # should only execute at the end of sequence
    def on_predict_end(self, logs=None):

        # old attn class, do not use until fixed
        #em = self.model.get_layer("v_self_attn").em.numpy()

        em = self.model.get_layer("attention").em.numpy()
        em = em.flatten()

        for value in em:
            self.fout1.write(str(value) + "\n")

        if np.min(em) == np.max(em):
            seq_norm = np.full([cfg.SEQLENGTH], np.max(em))
        else:
            seq_norm = (em - np.min(em)) / (np.max(em) - np.min(em))

        for value in seq_norm:
            self.fout2.write(str(value)+"\n")

        self.fout1.close()
        self.fout2.close()
