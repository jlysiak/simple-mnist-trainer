# MNISTrainer
## Assignment 1 - DNN @ MIMUW 2017/18

JACEK ŁYSIAK

### Requirements

In general, virtualenv with python3.5 is required.  
Package dependencies are listed in `requirements.txt`.  
Run `make requirements` in activated virtualenv to install.  

It's obvious, but to run training and visualization you   
must provide MNIST dataset (see notes below).  


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

#### NOTE 
If `-c` flat is not provided, checkpoints are saved in `/tmp` directory.  
If provided directory doesn't exist, script will create it.
If not, trainer tries load checkpoint from there, but if fails,
will start whole training from the begining (just run global initializer.)

If you provide wrong MNIST path, TensorFlow will download it.

