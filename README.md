# Next Word Prediction Using GRU RNN

This project demonstrates a next-word prediction model built using a Gated Recurrent Unit (GRU) Recurrent Neural Network. The model is trained on real-world textual data scraped from Wikipedia and learns to predict the next word in a sequence.

## 📝 Overview

The project leverages deep learning to generate context-aware next-word predictions using a single-layer GRU model. The training corpus is derived from the Wikipedia page on "Cat", which is cleaned, tokenized, and converted into input-output pairs suitable for sequence modeling.

## 🌐 Data Collection

- Source: [Wikipedia - Cat](https://en.wikipedia.org/wiki/Cat)
- Method: Scraped using `requests` and `BeautifulSoup`
- Preprocessing:
  - Lowercasing
  - Removal of punctuation and digits
  - Whitespace normalization

## 📊 Data Preparation

- Tokenization using Keras `Tokenizer`
- Input sequences are created using a sliding window over the tokenized corpus
- Sequences are padded to uniform length
- Labels are one-hot encoded

## 🧠 Model Architecture

- **Embedding Layer**: Transforms word indices to dense vectors
- **GRU Layer**: Learns temporal dependencies in sequences
- **Dense Output Layer**: Predicts the next word using softmax activation

```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_len-1),
    GRU(150),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])
