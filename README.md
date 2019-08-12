## RNN, LSTM and PyTorch


## Recurrent Neural Network Implementation based on WildML's tutorial
-------------------------------------------------------------------
The repository provides an implementation of Recurrent Neural Network based on tutorial [http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/).

Relevant files are:
- RNN.py
- RNN_impl.py
- rnn_utils.py

The code has been written in the form of the package that can be used with `import` command.

## Preparing your system/Installation

### CUDA Installation
Since I am using Ubuntu 18.04, I will install appropriate version of CUDA for my Ubuntu 18.04 system. Depending on your system, you can choose write CUDA package for your system from [http://developer.download.nvidia.com/compute/cuda/repos/](http://developer.download.nvidia.com/compute/cuda/repos/)


```bash
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.168-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804_10.1.168-1_amd64.deb

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo apt-get update
sudo apt-get install -y cuda

```

At this point, you will need to restart your computer. Then set the environment variable by including them in your `.bashrc` file.

```bash
# Set Environment variables
export CUDA_ROOT=/usr/local/cuda-10.1
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
# For profiling only
export CUDA_LAUNCH_BLOCKING=1
```

## Long Short-term Memory
----------------------

An example code is given to predict crypto currency exchange rate using LSTM. 

Relevant file(s):
crypt_lstm.py

## Using PyTorch for LSTM, Seq2Seq Model
----------------------------------------


A Juypter Notebook `PyTorch_1.ipynb` follows the tutorial from YouTube tutorial [Applied Deep Learning with PyTorch](https://www.youtube.com/watch?v=CNuI8OWsppg&t=19761s) to demonstrate the use of PyTorch to implement Seq2Seq Model.

