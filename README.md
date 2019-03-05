# Arabic to English Machine Translation with Google Transformer Model
This is an implementation of Machine Translation from Arabic to English using the [Transformer Model](https://arxiv.org/abs/1706.03762). 
It's Based on the [code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) provided by the authors.

We train the model using the [OpenSubtitles V2018](http://opus.nlpl.eu/OpenSubtitles-v2018.php) arabic-english parallel dataset.


## Walkthrough

Below are the commands for running the Transformer model. See the [Detailed instrutions](#detailed-instructions) for more details on running the model.

```
cd /path/to/models/official/transformer

# Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
# export PYTHONPATH="$PYTHONPATH:/path/to/models"

# Export variables
PARAM_SET=big
DATA_DIR=$HOME/transformer/data
MODEL_DIR=$HOME/transformer/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

# Download training/evaluation datasets
python data_download.py --data_dir=$DATA_DIR

# Train the model for 10 epochs, and evaluate after every epoch.
python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --bleu_source=test_data/newstest2014.en --bleu_ref=test_data/newstest2014.de

# Run during training in a separate process to get continuous updates,
# or after training is complete.
tensorboard --logdir=$MODEL_DIR

# Translate some text using the trained model
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --text="hello world"

# Compute model's BLEU score using the newstest2014 dataset.
python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --file=test_data/newstest2014.en --file_out=translation.en
python compute_bleu.py --translation=translation.en --reference=test_data/newstest2014.de
```

### Preprocessing
Before training, we strip all Tashkeel from the arabic sentences in the OpenSubtitles dataset using [pyarabic](https://pypi.org/project/PyArabic/)
### Training

For now, we only trained a slightly modified version of the `tiny` model with the following hyperparameters:
```
num_hidden_layers=6,
hidden_size=64,
num_heads=4,
filter_size=256,

layer_postprocess_dropout=0.1,
attention_dropout=0.1,
relu_dropout=0.1,

optimizer_adam_beta1=0.9,
optimizer_adam_beta2=0.997,
optimizer_adam_epsilon=1e-09
```

![training loss](/img/loss.png)
#### Time
Model was trained for ~20h for 10 epochs with almost 2 hours/epoch


### Evaluation results
We sampled a 1000-sentence portion from the OpenSubtitles v2018 training set for evaluation.
Below are the case-insensitive BLEU scores after 10 epochs.

Param Set | Score
--- | --- |
tiny | 26.54

![evaluation bleu](/img/bleu.png)



