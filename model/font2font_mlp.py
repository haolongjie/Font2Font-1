import os
import tensorflow as tf


# Network parameters
n_input = 256*256*3
n_hidden_1 = 1000
n_hidden_2 = 1000
n_hidden_3 = 1000
n_hidden_4 = 1000
n_hidden_5 = 1000
n_hidden_6 = 1000
n_output = 256*256*3


class Font2Font(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0, input_filters=3, output_filters=3):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.input_filters = input_filters
        self.output_filters = output_filters
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def build_model(self, is_training=True):
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_output])

        # H1
        w1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
        b1 = tf.Variable(tf.random_normal([n_hidden_1]))
        h1 = tf.add(tf.matmul(x, w1), b1)
        h1 = tf.nn.relu(h1)
        # H2
        w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
        b2 = tf.Variable(tf.random_normal([n_hidden_2]))
        h2 = tf.add(tf.matmul(h1, w2), b2)
        h2 = tf.nn.relu(h2)
        # H3
        w3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
        b3 = tf.Variable(tf.random_normal([n_hidden_3]))
        h3 = tf.add(tf.matmul(h2, w3), b3)
        h3 = tf.nn.relu(h3)






    def train(self):
        pass

    def test(self):
        pass