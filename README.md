# Yelp Sentiment Analysis (FFNN & RNN)

This project implements Feedforward and Recurrent Neural Networks for 5-class sentiment analysis on Yelp reviews.

---

## FFNN

Bag-of-words input → hidden layer (ReLU) → output layer → LogSoftmax. Uses SGD optimizer and fixed epochs.

## RNN

Sequence of word embeddings → RNN (tanh) → linear output per token → sum over sequence → LogSoftmax. Uses Adam optimizer, pretrained embeddings from ``word_embedding.pkl``, and early stopping with max-epoch cap.

---

## Running the code

**FFNN**

``python ffnn.py -hd 64 -e 10 --train_data training.json --val_data validation.json``

**RNN**

``python rnn.py -hd 64 -e 50 --train_data training.json --val_data validation.json``

(Ensure ``word_embedding.pkl`` is in the project root or adjust the path in rnn.py.)

