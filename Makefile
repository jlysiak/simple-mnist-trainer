
MODEL="model_ckpt"
IMGS=images
LOG=train.log
DS=MNIST_data


requirements:
	pip install -r requirements.txt

train:
	python3 mnist.py -M $(DS) -l $(LOG) -c $(MODEL) -t

images:
	python3 mnist.py -m $(DS) -i $(IMGS) -c $(MODEL) -v

clean:
	rm -fr $(MODEL) $(IMGS) $(LOG)
