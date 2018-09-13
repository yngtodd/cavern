import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "1" and/or "0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
#from keras.models import load_model
from keras.callbacks import Callback, ModelCheckpoint

from lstm_vae import create_lstm_vae

import argparse
from skopt import load
from skopt import dump
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver

# define parameters
nb_start = 0
nb_end = 40
frame = int(224000*0.03)
timesteps = 100
intermediate_dim = 32
latent_dim = 3
epochs = 25
batch_size = 1
epsilon_std = 1.
data_file = "fs-peptide_encoded_train_150_tf.npy"
data_path = "./data/fs-peptide/"
result_path = "./result/"
loss_path = "./result/loss_fs-peptide/"
model_path = "./result/model_fs-peptide/"

# define hisotry for loss
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        #self.val_losses = []
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #self.val_losses.append(logs.get('val_loss'))
history = LossHistory()

# define how to accomodate data
def get_data():
    # read data from file
    data = np.load(data_path + data_file)
    dataX = []
    for i in range(len(data[:frame]) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)

# main LSTM autoencoder
x = get_data()
input_dim = x.shape[-1]
timesteps = x.shape[1]

def objective(params):
    encode_l1, encode_dropout, decode_l1, decode_dropout, intermediate_dim, latent_dim = params
    # load model
    vae, enc, gen = create_lstm_vae(input_dim,
        timesteps=timesteps,
        batch_size=batch_size,
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        epsilon_std=epsilon_std)

    # train model
    vae.fit(x, x, epochs=epochs, callbacks=[history])
    return np.mean(history.losses)


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    hparams = [(0.001, 0.1),
               (0.0, 0.90),
               (0.001, 0.1),
               (0.0, 0.90),
               (16, 32),
               (3, 13)]

    res = load('./checkpoint.pkl')

    x0 = res.x_iters
    y0 = res.func_vals
    print(f'Previous iteration made it through {len(y0)} iterations')

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl")
    res_gp = gp_minimize(objective, hparams, x0=x0, y0=y0, n_calls=20, callback=[checkpoint_saver], random_state=0, verbose=True)
    dump(res_gp, 'smbo20')

if __name__ == "__main__":
    main()
