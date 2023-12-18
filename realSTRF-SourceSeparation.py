import argparse
import glob
import time
import librosa
import pickle

import numpy as np
from math import floor, ceil
from torch.utils import data
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap, config, random
config.update("jax_enable_x64", True)

import optax
import flax.linen as nn
from random import Random

from strfpy import *
from strfpy_jax import *
from model import *
from dataset import *
print(f"Training on {jax.default_backend()}...")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_strfs', required=True, type=int) 
  parser.add_argument('--strf_seed', default=0, type=int) 
  parser.add_argument('--spec_type', required=True, type=str, help='3 options: logMel, logAud, linAud.')
  parser.add_argument('--weight_norm', required=True, type=int)
  parser.add_argument('--conv_features', required=True, nargs="+", type=int)
  parser.add_argument('--activation_fct', required=True, nargs="+", type=int)
  parser.add_argument('--sigmoid_last', required=True, type=int)

  parser.add_argument('--lr_v', required=True, type=float)
  parser.add_argument('--lr_sr', required=True, type=float)
  parser.add_argument('--num_steps', default=200000, type=int)
  parser.add_argument('--save_step', default=10000, type=int)
  parser.add_argument('--minibatch_size', default=4, type=int)

  parser.add_argument('--home_dir', required=True, type=str)
  parser.add_argument('--clean_dir', required=True, type=str)
  parser.add_argument('--noise_dir', required=True, type=str)
  parser.add_argument('--snr', required=True, type=float)
  parser.add_argument('--save_dir', required=True, type=str)

  config = parser.parse_args()
  Bs, As = read_cochba_j()

  @jit
  def wav2aud_lin(x):
    '''Output size: 200 x 128'''    
    return wav2aud_j(x, 5, 8, -2, 0, As, Bs, fft=True, return_stage=5).T
  batch_wav2aud_lin = vmap(wav2aud_lin)

  @jit
  def wav2aud_log(x, log_spec, pos):
    '''Output size: 200 x 128'''
    out = wav2aud_j(x, 5, 8, -2, 0, As, Bs, fft=True, return_stage=5).T
    eps = 1e-10
    out = jnp.log(jnp.array(out)+eps)
    out += -jnp.log(eps)
    #out = (out-jnp.mean(out))/jnp.var(out)
    return out
  batch_wav2aud_log = vmap(wav2aud_log)
  
  def batch_melspec(x):
    '''Output size: 4 x 200 x 128'''
    s = []
    if log_spec==0: 
      raise NotImplementedError
    eps = 1e-10
    for i in range(len(x)):
      temp = librosa.feature.melspectrogram(y=np.array(x[i,:]), sr=16000, n_fft=512, hop_length=80)[:,:200].T
      temp = jnp.log(jnp.array(temp)+eps)
      temp += -jnp.log(eps)
      #temp = (temp-jnp.mean(temp))/jnp.var(temp)
      s.append(temp)
    return jnp.stack(s)

  # Can this be replaced by a generic L2 loss?
  def audspec_loss(s, s_hat, loss='L2'):
    if loss=='L2':
        return jnp.mean((s-s_hat)**2)
    else: 
        raise NotImplementedError
  batch_audspec_loss = vmap(audspec_loss)

  @jit
  def forward_loss_batch(variables, sn, sc, sr):
    '''shape of input: batch is in the first (0) dimension'''
    s_hat = model.apply(variables, sn, sr)
    return jnp.mean(batch_audspec_loss(sc, s_hat))
  
  def update_batch_sr(variables, sr, ov, osr, opt_state_v, opt_state_sr, xn, xc): 
    if config.spec_type=='logAud':
      sn = batch_wav2aud_log(xn)
      sc = batch_wav2aud_log(xc)
    elif config.spec_type=='linAud':
      sn = batch_wav2aud_lin(xn)
      sc = batch_wav2aud_lin(xc)
    elif config.spec_type=='logMel':
      sn = batch_melspec(xn)
      sc = batch_melspec(xc)
    else: raise NotImplementedError
    loss, (g_v, g_sr) = value_and_grad(forward_loss_batch, (0, 3))(variables, sn, sc, sr)
    u_v, opt_state_v = ov.update(g_v, opt_state_v)
    u_sr, opt_state_sr = osr.update(g_sr, opt_state_sr)
    return optax.apply_updates(variables, u_v), optax.apply_updates(sr, u_sr), opt_state_v, opt_state_sr, loss

  model = vAudioSTRFAE(weight_norm=config.weight_norm, conv_features=config.conv_features, 
                       activation_fct=config.activation_fct, sigmoid_last=config.sigmoid_last)
  
  sr = initialize_strfs(config.num_strfs, scale_cap=8, rate_cap=28, seed=config.strf_seed)
  variables = model.init(jax.random.key(0), 
                         jnp.ones([config.minibatch_size, 200, 128]), sr)

  train_set = CocktailAudioDataset(home_dir=config.home_dir, clean_dir=config.clean_dir, noise_dir=config.noise_dir, 
                              sampling=1., snr=config.snr)
  sampler = data.RandomSampler(train_set, replacement=True, 
                               num_samples=config.minibatch_size*config.num_steps)
  train_data_loader = data.DataLoader(train_set, batch_size=config.minibatch_size, sampler=sampler)
  # TODO: add validation set
  
  op_v, op_sr = optax.adam(config.lr_v), optax.adam(config.lr_sr)
  opt_state_v = op_v.init(variables)
  opt_state_sr = op_sr.init(sr)
  print(f"The training set has {len(train_set)} utterances.")
  print(f'Number of STRFs: {config.num_strfs}')
  t0 = time.time()
  total_loss = 0
  step = 1
  with open(config.save_dir+'_chkStep'+str(step)+'.p', 'wb') as temp:
    pickle.dump([variables, sr], temp)
  for xn, xc in train_data_loader:
    xn, xc = jnp.array(xn), jnp.array(xc)
    variables, sr, opt_state_v, opt_state_sr, l = update_batch_sr(
      variables, sr, op_v, op_sr, opt_state_v, opt_state_sr, xn, xc)
    total_loss += l
    
    if step % config.save_step == 0:
      with open(config.save_dir+'.log', 'a') as temp:
        temp.write(f"Step: {step}; loss: {total_loss/config.save_step}; Time: {(time.time()-t0)/60} min.\n")
      with open(config.save_dir+'_chkStep'+str(step)+'.p', 'wb') as temp:
        pickle.dump([variables, sr], temp)
      t0 = time.time()
      total_loss = 0
    step += 1

import pickle
from functools import partial

from math import ceil
from jax import jit
from jax.image import resize 
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from random import Random

class audioSTRFAE(nn.Module):
  """Convolutional Decoder."""
  weight_norm: int
  conv_features: int
  activation_fct: int
  sigmoid_last: int

  def __call__(self, x, sr):
    return self.conv(self.encode(x, sr))

  def encode(self, spec, sr_orig):
    '''Use STRFs to encode stimuli'''
    sr = jnp.abs(sr_orig)
    out = strf(spec, sr).real
    out = out.transpose(2, 1, 0)
    return out

  @nn.compact
  def conv(self, x):
    '''Input: frequency (128) x time (200/s) x n_channel (e.g. 198)'''
    n_layers = len(self.conv_features)
    for i in range(n_layers):
      if self.weight_norm == 1:
        x = nn.WeightNorm(nn.Conv(features=self.conv_features[i], kernel_size=(3, 3), strides=(1,1)), use_scale=False, feature_axes=2)(x) 
      else:
        x = nn.Conv(features=self.conv_features[i], kernel_size=(3, 3), strides=(1,1))(x) 
      if self.activation_fct[i]==1:
        if self.sigmoid_last==1 and i==n_layers-1:
          x = nn.relu(x)
        else:
          x = nn.gelu(x)
    x = x.squeeze().T
    return x

def initialize_strfs(n_strfs, seed=0, scale_cap=9, rate_cap=9, proportion_negative_signs=0.5):
  # Between 0 and 1
  np.random.seed(seed)
  s, v = np.random.rand(n_strfs)*scale_cap, np.random.rand(n_strfs)*rate_cap
  return np.stack([s,v]).T

vAudioSTRFAE = nn.vmap(
    audioSTRFAE,
    in_axes=(0, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    methods=["__call__", "conv", "encode"])

if __name__ == '__main__':
  main() 
