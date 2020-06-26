# What's the point of this repo?

In doing frame classification for news, augmenting news headlines with information about the image
accoompaniying the neadline helps improve performance.

The "information about the image" comes from Google's Web Detection API for images.

# What does the data look like? 

In `data_subsets/all_data/`, there are five directories, corresponding to five folds of the same 1300 
news headlines split into training and validation sets. Each is a CSV file. The files contain a lot of
extra metadata we don't use, so we present below an excerpt of the CSV file, where unimportant columns 
have been truncated.

| news_title                                                                                        | Q3 Theme1 | WebDetectEntities                                                                                                                                                                             |
|---------------------------------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cremated remains of Las Vegas mass shooter to be kept in safe deposit box, brother says"          | 7         | "Stephen Paddock, 2017 Las Vegas Strip shooting, Shooting, Motive, Mass shooting, McCarran International Airport, Firearm, Eric Paddock, Massacre, Las Vegas Metropolitan Police Department"  |
| Florida shooter a troubled loner with white supremacist ties                                      | 4         | "Nikolas Cruz, Stoneman Douglas High School shooting, Marjory Stoneman Douglas High School, School shooting, Shooting, Murder, Mass shooting, AR-15 style rifle, Suspect, Student"            |
| Vernon Hills teen accused of wearing white supremacist shirt pleads not guilty to weapons charges | 6         | "Vernon Hills, FOID, Firearm, Ammunition, White supremacy, Bureau of Alcohol, Tobacco, Firearms and Explosives, Gun, Neo-Nazism, Arrest"                                                      |
| Griffith student charged with accidentally bringing loaded gun inside high school                 | 5         | "Arrest, Student, Expulsion, School, Felony, Ada County, Mug shot, Criminal charge, Police, Driving under the influence"                                                                      |
| "Exclusive: Group chat messages show school shooter obsessed with race, violence and guns"        | 4         | "Nikolas Cruz, Stoneman Douglas High School shooting, Marjory Stoneman Douglas High School, School shooting, Shooting, Rifle, Broward County Sheriff's Office, School, Murder, Mass shooting" |

I hope the column headers are descriptive enough. 

# Ok, I saw the data. Now what?

In this repo, one can train a BERT model to predict the frame using (1) only the news headline, 
(2) only the web detection api results, or (3) both the headline and the web detection api results.

The results were that (3) yields better results than (1).

# Cool. But what do you *actually* provide as input to BERT in the three cases above?

 1. For news headline only: `[CLS]` + news headline + `[SEP]`
 2. For web detection API only: `[CLS]` + comma separated word detection results + `[SEP]`
 3. For news headlines and web detection: `[CLS]` +news headlines + `[SEP]` + web detection results + `[SEP]`

PS: I'm actually not sure if there's a `[SEP]` at the end of the input for option 3 above. But
I'm sure that this repo is doing the right thing because we offload this preparation to the excellent
`transformers` library.

# I get it now. Show me the code.

The part to pay attention to in this repo is the `processors.py` file. In there, there
are three "processors" that determine what the input will be to bert:

 2. `webdetectonly`: This feeds in the web detection results only to BERT.
 3. `frame`: This feeds in the the headlines only to BERT.
 1. `webdetect`: This feeds in the headlines + the web detection results to BERT.

This is a terrible naming scheme. I apologize.

# How do I run this code?

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
	--do_lower_case \
	--do_train \
	--do_eval
```
## I ran the code. But where are the results?

The [line here](https://github.com/davidatbu/newsframing_webdetection_api/blob/71452aa3529afebfbc63a99f8606c4457ee04a80/run#L11) specifies where the output of training (model weights, predictions, and metrics) will be.

In side that directory, the `run` script that you ran will create a directory named according to what you passed in as
`--dataset_name`. In our case it was `all_data`. 

The output will look something like this.

	all_data/
	└── 13:41,26_06_20 # This is a time stamp of when you started the training
	    ├── 0                                # There will be a directory for each fold
	    │   ├── config.json                      # transformers library own config 
	    │   ├── eval_results-all_data.json       # f1, accuracy, etc on validation set
	    │   ├── logits.csv                       # logits predicted for validation set
	    │   ├── pytorch_model.bin                
	    │   ├── special_tokens_map.json
	    │   ├── tokenizer_config.json
	    │   ├── training_args.bin
	    │   └── vocab.txt
	    ├── 1
	    │   ...
	    ├── 2
	    │   ...
	    ├── 3
	    │   ...
	    ├── 4
	    │   ...
	    └── args.json # THis contains the arguments that you passed to the `run` script above.
