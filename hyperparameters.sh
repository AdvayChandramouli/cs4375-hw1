#!/bin/bash
# FFNN experiments
python ffnn.py -hd 32 -e 5 --train_data training.json --val_data validation.json
python ffnn.py -hd 64 -e 10 --train_data training.json --val_data validation.json

# RNN experiments
python rnn.py -hd 32 -e 50 --train_data training.json --val_data validation.json
python rnn.py -hd 64 -e 50 --train_data training.json --val_data validation.json
