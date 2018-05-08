# MNISTrainer
## Assignment 1 - DNN @ MIMUW 2017/18

JACEK ŁYSIAK

### Requirements

In general, virtualenv with python3.5 is required.
Package dependencies are listed in `requirements.txt`.
Run `make requirements` in active virtualenv to install.

(It's obvious, but...) To run training and visualization you must provide MNIST dataset.

### Running

Quick commands are provided in `Makefile`.
Run `make train` to start training model provided in `model_ckpt`.
Run `make images` to generate images which maximize values of filters in 
convolutional layers.

If you want to modify layers of neural network, edit `LAYERS_CONF` list at
the begining of `mnist.py`.

`mnist.py` script requires some flags.  
To see description run `python3.5 mnist.py -h` but, in short:  

Obligatory flags: 
  * `-t` - training | `-v` - visualization
  * `-M DIR` - MNIST dataset location (e.g whole directory `MNIST_data` from Dropbox)  

Extra flags:
  * `-c CKPT_DIR` - save/load checkpoint from here
  * `-l LOG_FILE` - save some train logs here
  * `-i IMGS_DIR` - save images here

