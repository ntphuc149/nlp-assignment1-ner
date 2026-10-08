# NLP Assignment 1 - NER on WNUT-2016 (115522605)

BiLSTM/GRU/RNN + GloVe-Twitter + char-CNN + casing features + CRF, implemented in PyTorch (`115522605.py`).

## Run on Kaggle
Enable GPU and Internet (Settings), then in a notebook:
```
!git clone https://github.com/ntphuc149/nlp-assignment1-ner.git
%cd nlp-assignment1-ner
!python 115522605.py --mode ablation   # architecture comparison -> outputs/ablation.csv
!python 115522605.py --mode final      # ensemble -> outputs/result.txt, dev_results.json, training_curves.png
```
GloVe (`glove.twitter.27B.zip`) is downloaded automatically. If Internet is off, add a Kaggle
dataset containing GloVe-Twitter and it is auto-detected under `/kaggle/input`, or pass `--glove PATH`.

Useful flags: `--encoder {rnn,gru,lstm} --crf {0,1} --char {0,1} --case {0,1} --emb_dim {25,50,100,200}`.

## Submission
`<id>_hw1.zip` = `115522605.py`, `result.txt` (from `outputs/`), `requirements.txt`
(`pip freeze > requirements.txt` on Kaggle), `115522605_report.pdf`.
