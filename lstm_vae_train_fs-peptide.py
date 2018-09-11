import sys, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "1" and/or "0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import Callback, ModelCheckpoint

from lstm_vae import create_lstm_vae 

# define parameters 
nb_start = 0 
nb_end = 40     
frame = int(224000*0.50)        
timesteps = 100 
intermediate_dim = 32 
latent_dim = 3
epochs = 1   
batch_size = 1     
epsilon_std = 1. 
data_file = "fs-peptide_encoded_train_150_tf.npy"
data_path = "./data/fs-peptide/"
result_path = "./result/"
loss_path = "./result/loss_fs-peptide/"
model_path = "./result/model_fs-peptide/"

# create directories;
if not os.path.exists(data_path):
   os.mkdir(data_path, 0755);
if not os.path.exists(result_path):
   os.mkdir(result_path, 0755);
if not os.path.exists(loss_path):
   os.mkdir(loss_path, 0755); 
if not os.path.exists(model_path):
   os.mkdir(model_path, 0755); 
print "directories created or if already exists - then checked";

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
if __name__ == "__main__":
    x = get_data()
    input_dim = x.shape[-1]
    timesteps = x.shape[1]
    # load model
    vae, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=intermediate_dim,
        latent_dim=latent_dim,
        epsilon_std=epsilon_std)
    # train & save & load 
    for i in range (nb_start, nb_end): 
    	if i == 0:
	   print("skipping - no previous saved file to load")
	# load saved model 
	else: 
	   vae.load_weights(model_path + './lstm_vae_%i' %i) 
	# train model  
	vae.fit(x, x, epochs=epochs, callbacks=[history])
	vae.save_weights(model_path + './lstm_vae_%i' %(i+1)) 
	# save loss values over epochs 
    	np.savetxt(loss_path + 'hist.losses_%i'%(i+1), history.losses, delimiter = ',')
    # compile loss value   
    hist = np.zeros(((nb_end-nb_start),2))   
    for i in range ((nb_start+1), (nb_end+1)):
    	hist_loss = np.loadtxt(loss_path + "hist.losses_%i" %i)     
    	tmp = np.array([i, hist_loss]) 
    	hist[i-1] = tmp       
    np.savetxt(loss_path + 'hist_tot', hist, delimiter=' ')    

    # predict & encode 
    #preds = vae.predict(x[:], batch_size=batch_size)
    #np.save("./pred.npy", preds)
    #encodes = enc.predict(x[:], batch_size=batch_size)
    #np.save("./enc.npy", encodes)




