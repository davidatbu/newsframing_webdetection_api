# Where the important part is

The part to pay attention to in this repo is the `processors.py` file. In there, there
are three "processors" that determine what the input will be to bert:

 1. `webdetect`: This feeds in the headlines + the web detection results to BERT.
 2. `webdetectonly`: This feeds in the web detection results only to BERT.
 3. `frame`: This feeds in the the headlines only to BERT.

This is a terrible naming scheme. I apologize.

# How to run

## Install prerequisites
Make sure to prepare your Python environment according to `requirements.txt`. After you change
the CUDA version of the PyTorch package in `requirements.txt` (currently it's set at 10.1) to whatever CUDA
version you have on your machine, do:

	pip install -r requirements.txt


## Actual command to run
Here's a slightly annotated bash command that you should be able to use.

```bash
./run \
	--task_name webdetect \   # This can be one of webdetect, webdetectonly, or frame. (check above)
	--dataset_name all_data \ # Specifies what directory to look for in data_subsets/
	--model_type bert \
	--model_name_or_path bert-base-uncased \
	--per_gpu_train_batch_size 4 \  # This, and the two lines below are critical to getting good performance on this dataset.
	--num_train_epochs 10 \
	--learning_rate 2e-5 \
	--max_seq_length 128 \
	--save_steps 9999 \
	--logging_steps 9999 \
	--do_train \
	--do_eval
```

