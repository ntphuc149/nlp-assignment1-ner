"""
NLP Assignment 1 - Named Entity Recognition on WNUT-2016 (student id: 115522605)

Model: (frozen) pre-trained GloVe-Twitter word embeddings
       + character CNN + casing/token-type embedding
       -> (Bi)RNN / GRU / LSTM encoder -> softmax or CRF output layer.
The CRF, the BIO-constrained Viterbi decoder and the entity-level (CoNLL-style)
P/R/F1 are implemented from scratch in this file; only PyTorch/NumPy are required.

Usage
-----
  python 115522605.py --mode final      # train ensemble, write outputs/result.txt
  python 115522605.py --mode ablation   # compare architectures, write outputs/ablation.csv
  python 115522605.py --mode quick      # one fast run (sanity check)

GloVe is searched in --glove, /kaggle/input/**, ./glove, and otherwise downloaded
(glove.twitter.27B.zip, ~1.4GB, read directly from the zip).
"""
import argparse
import copy
import csv
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
GLOVE_URLS = [
    "https://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip",
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip",
]
PAD, UNK = 0, 1
MAX_WORD_LEN = 20


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="final", choices=["final", "ablation", "quick"])
    p.add_argument("--data_dir", default=os.path.join(HERE, "wnut_16"))
    p.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    p.add_argument("--glove", default=None, help="path to glove txt/zip")
    p.add_argument("--emb_dim", type=int, default=200, choices=[25, 50, 100, 200])
    p.add_argument("--encoder", default="lstm", choices=["rnn", "gru", "lstm"])
    p.add_argument("--bidirectional", type=int, default=1)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--char", type=int, default=1, help="use char-CNN")
    p.add_argument("--case", type=int, default=1, help="use casing / token-type feature")
    p.add_argument("--crf", type=int, default=1, help="CRF output layer (else softmax)")
    p.add_argument("--finetune", type=int, default=0, help="fine-tune GloVe vectors")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--word_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_models", type=int, default=5, help="ensemble size (final mode)")
    p.add_argument("--n_seeds", type=int, default=3, help="seeds per config (ablation)")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_conll(path, has_tags=True):
    """Returns list of (tokens, tags). tags is None for unlabeled files."""
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                if toks:
                    sents.append((toks, tags if has_tags else None))
                toks, tags = [], []
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            toks.append(parts[0])
            if has_tags:
                tags.append(parts[-1])
    if toks:
        sents.append((toks, tags if has_tags else None))
    return sents


URL_RE = re.compile(r"^(https?://|www\.)", re.I)
NUM_RE = re.compile(r"^[-+]?[\d.,:/]*\d[\d.,:/]*$")
CASE_NAMES = ["pad", "lower", "upper", "title", "mixed", "digit", "user", "url", "hashtag", "punct", "other"]
CASE2ID = {n: i for i, n in enumerate(CASE_NAMES)}


def token_type(tok):
    if tok.startswith("@") and len(tok) > 1:
        return "user"
    if URL_RE.match(tok):
        return "url"
    if tok.startswith("#") and len(tok) > 1:
        return "hashtag"
    if NUM_RE.match(tok):
        return "digit"
    if not any(c.isalnum() for c in tok):
        return "punct"
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return "other"
    if all(c.isupper() for c in letters) and len(letters) > 1:
        return "upper"
    if tok[0].isupper() and all(c.islower() for c in letters[1:]):
        return "title"
    if all(c.islower() for c in letters):
        return "lower"
    return "mixed"


def glove_candidates(tok):
    """Normalisation (GloVe-Twitter style) -> ordered lookup keys."""
    t = token_type(tok)
    if t == "user":
        return ["<user>"]
    if t == "url":
        return ["<url>"]
    if t == "digit":
        return ["<number>", tok.lower()]
    low = tok.lower()
    if t == "hashtag":
        return [low[1:], low]
    squeezed = re.sub(r"(.)\1{2,}", r"\1\1", low)  # sooooo -> soo
    cands = [low]
    if squeezed != low:
        cands.append(squeezed)
    if squeezed != re.sub(r"(.)\1+", r"\1", squeezed):
        cands.append(re.sub(r"(.)\1+", r"\1", squeezed))
    return cands


def find_glove(args):
    dim = args.emb_dim
    name = f"glove.twitter.27B.{dim}d.txt"
    cands = [args.glove] if args.glove else []
    cands += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    cands += glob.glob(f"/kaggle/input/**/glove.twitter.27B.zip", recursive=True)
    cands += [os.path.join(HERE, "glove", name), os.path.join(HERE, "glove", "glove.twitter.27B.zip"),
              os.path.join(args.out_dir, "glove.twitter.27B.zip")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, "glove.twitter.27B.zip")
    for url in GLOVE_URLS:
        try:
            print(f"Downloading GloVe from {url} ...", flush=True)
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:  # noqa
            print("  failed:", e)
    sys.exit("Could not obtain GloVe. Pass --glove /path/to/glove.twitter.27B.{dim}d.txt (or the .zip).")


def load_glove(path, dim, needed):
    vecs = {}

    def consume(fh):
        for line in fh:
            if isinstance(line, bytes):
                line = line.decode("utf-8", "ignore")
            word, _, rest = line.partition(" ")
            if word in needed:
                v = np.asarray(rest.split(), dtype=np.float32)
                if v.shape[0] == dim:
                    vecs[word] = v

    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            member = [n for n in z.namelist() if n.endswith(f"glove.twitter.27B.{dim}d.txt")][0]
            with z.open(member) as fh:
                consume(fh)
    else:
        with open(path, encoding="utf-8") as fh:
            consume(fh)
    return vecs


class Data:
    """Vocabularies + tensorised splits."""

    def __init__(self, args):
        d = args.data_dir
        self.train = read_conll(os.path.join(d, "train.txt"))
        self.dev = read_conll(os.path.join(d, "dev.txt"))
        self.test = read_conll(os.path.join(d, "test.txt"), has_tags=False)

        tags = sorted({t for _, ts in self.train + self.dev for t in ts}, key=lambda x: (x != "O", x))
        self.tag2id = {t: i for i, t in enumerate(tags)}
        self.id2tag = tags

        # word vocab: every token (from all splits; no labels used) that has a GloVe vector
        all_tokens = {tok for ss in (self.train, self.dev, self.test) for toks, _ in ss for tok in toks}
        needed = {c for tok in all_tokens for c in glove_candidates(tok)}
        gpath = find_glove(args)
        print(f"Loading GloVe {args.emb_dim}d from {gpath}", flush=True)
        vecs = load_glove(gpath, args.emb_dim, needed)
        self.word2id, rows = {"<pad>": PAD, "<unk>": UNK}, [np.zeros(args.emb_dim), np.zeros(args.emb_dim)]
        self.tok2key = {}
        for tok in all_tokens:
            for c in glove_candidates(tok):
                if c in vecs:
                    self.tok2key[tok] = c
                    if c not in self.word2id:
                        self.word2id[c] = len(rows)
                        rows.append(vecs[c])
                    break
        self.emb = np.stack(rows).astype(np.float32)
        cov = np.mean([t in self.tok2key for t in all_tokens])
        print(f"GloVe vocab={len(self.word2id)}  token-type coverage={cov:.1%}", flush=True)

        chars = Counter(c for tok in all_tokens for c in tok)
        self.char2id = {"<pad>": 0, "<unk>": 1}
        for c in chars:
            self.char2id[c] = len(self.char2id)

        self.train_t = [self.encode(s) for s in self.train]
        self.dev_t = [self.encode(s) for s in self.dev]
        self.test_t = [self.encode(s) for s in self.test]

    def encode(self, sent):
        toks, tags = sent
        w = [self.word2id.get(self.tok2key.get(t), UNK) for t in toks]
        c = [[self.char2id.get(ch, 1) for ch in t[:MAX_WORD_LEN]] for t in toks]
        k = [CASE2ID[token_type(t)] for t in toks]
        y = [self.tag2id[t] for t in tags] if tags is not None else [0] * len(toks)
        return w, c, k, y


def collate(items, device):
    B, T = len(items), max(len(i[0]) for i in items)
    L = max(len(c) for i in items for c in i[1])
    w = torch.zeros(B, T, dtype=torch.long)
    c = torch.zeros(B, T, L, dtype=torch.long)
    k = torch.zeros(B, T, dtype=torch.long)
    y = torch.zeros(B, T, dtype=torch.long)
    lengths = torch.tensor([len(i[0]) for i in items])
    for b, (wi, ci, ki, yi) in enumerate(items):
        n = len(wi)
        w[b, :n], k[b, :n], y[b, :n] = torch.tensor(wi), torch.tensor(ki), torch.tensor(yi)
        for t, cc in enumerate(ci):
            c[b, t, : len(cc)] = torch.tensor(cc)
    return w.to(device), c.to(device), k.to(device), y.to(device), lengths


def batches(items, bs, shuffle):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i : i + bs], [items[j] for j in idx[i : i + bs]]


# ----------------------------------------------------------------------------
# Evaluation (entity-level exact match, conlleval semantics)
# ----------------------------------------------------------------------------
def get_chunks(tags):
    chunks, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O":
            tag, ct = "O", None
        else:
            tag, ct = t.split("-", 1)
        end_prev = typ is not None and (tag in ("O", "B") or ct != typ)
        if end_prev:
            chunks.add((typ, start, i - 1))
            typ = None
        if tag == "B" or (tag == "I" and typ is None):
            start, typ = i, ct
    return chunks


def evaluate(gold_seqs, pred_seqs):
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold_seqs, pred_seqs):
        gc, pc = get_chunks(g), get_chunks(p)
        for ch in gc & pc:
            tp[ch[0]] += 1
        for ch in pc - gc:
            fp[ch[0]] += 1
        for ch in gc - pc:
            fn[ch[0]] += 1

    def prf(a, b, c):
        pr = a / (a + b) if a + b else 0.0
        rc = a / (a + c) if a + c else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    types = sorted(set(tp) | set(fp) | set(fn))
    res = {"per_type": {t: dict(zip(("p", "r", "f1"), prf(tp[t], fp[t], fn[t])), support=tp[t] + fn[t]) for t in types}}
    res["p"], res["r"], res["f1"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return res


# ----------------------------------------------------------------------------
# CRF / constrained Viterbi
# ----------------------------------------------------------------------------
def bio_masks(id2tag):
    """Additive masks forbidding illegal BIO moves (O->I-x, B-x->I-y, start->I-x)."""
    K = len(id2tag)
    trans = torch.zeros(K, K)
    start = torch.zeros(K)
    for j, tj in enumerate(id2tag):
        if tj.startswith("I-"):
            start[j] = -1e4
            for i, ti in enumerate(id2tag):
                if ti == "O" or ti[2:] != tj[2:]:
                    trans[i, j] = -1e4
    return trans, start


def viterbi(emis, lengths, trans, start, end):
    """emis: B x T x K (scores), returns list of tag-id lists."""
    B, T, K = emis.shape
    score = start + emis[:, 0]
    back = []
    for t in range(1, T):
        cand = score.unsqueeze(2) + trans.unsqueeze(0)  # B x K(prev) x K(cur)
        best, idx = cand.max(1)
        new = best + emis[:, t]
        m = (t < lengths.to(emis.device)).unsqueeze(1)
        score = torch.where(m, new, score)
        back.append(torch.where(m, idx, torch.arange(K, device=emis.device).expand(B, K)))
    score = score + end
    last = score.argmax(1)
    paths = torch.zeros(B, T, dtype=torch.long, device=emis.device)
    paths[:, T - 1] = last
    cur = last
    for t in range(T - 2, -1, -1):
        cur = back[t].gather(1, cur.unsqueeze(1)).squeeze(1)
        paths[:, t] = cur
    # for shorter sequences the identity back-pointers keep the final tag constant over padding
    return [paths[b, : lengths[b]].tolist() for b in range(B)]


class CRF(nn.Module):
    def __init__(self, K, mask_trans, mask_start):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(K, K))
        self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))
        self.register_buffer("mask_trans", mask_trans)
        self.register_buffer("mask_start", mask_start)

    def params(self):
        return self.trans + self.mask_trans, self.start + self.mask_start, self.end

    def nll(self, emis, tags, mask):
        trans, start, end = self.params()
        B, T, K = emis.shape
        ar = torch.arange(B, device=emis.device)
        gold = start[tags[:, 0]] + emis[ar, 0, tags[:, 0]]
        alpha = start + emis[:, 0]
        for t in range(1, T):
            m = mask[:, t]
            gold = gold + m * (trans[tags[:, t - 1], tags[:, t]] + emis[ar, t, tags[:, t]])
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0) + emis[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(m.bool().unsqueeze(1), nxt, alpha)
        last = tags[ar, mask.sum(1).long() - 1]
        gold = gold + end[last]
        logz = torch.logsumexp(alpha + end, dim=1)
        return (logz - gold).sum() / B


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Tagger(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        K = len(data.id2tag)
        self.word = nn.Embedding.from_pretrained(torch.tensor(data.emb), freeze=not args.finetune, padding_idx=PAD)
        self.unk_vec = nn.Parameter(torch.randn(args.emb_dim) * 0.1)
        in_dim = args.emb_dim
        if args.char:
            self.char = nn.Embedding(len(data.char2id), 30, padding_idx=0)
            self.char_cnn = nn.Conv1d(30, 50, kernel_size=3, padding=1)
            in_dim += 50
        if args.case:
            self.case = nn.Embedding(len(CASE_NAMES), 16, padding_idx=0)
            in_dim += 16
        self.drop = nn.Dropout(args.dropout)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[args.encoder]
        kw = {"nonlinearity": "tanh"} if args.encoder == "rnn" else {}
        self.rnn = rnn_cls(in_dim, args.hidden, num_layers=args.layers, batch_first=True,
                           bidirectional=bool(args.bidirectional),
                           dropout=args.dropout if args.layers > 1 else 0.0, **kw)
        out_dim = args.hidden * (2 if args.bidirectional else 1)
        self.proj = nn.Sequential(nn.Linear(out_dim, args.hidden), nn.Tanh(), nn.Dropout(args.dropout))
        self.out = nn.Linear(args.hidden, K)
        mt, ms = bio_masks(data.id2tag)
        if args.crf:
            self.crf = CRF(K, mt, ms)
        else:
            self.register_buffer("mask_trans", mt)
            self.register_buffer("mask_start", ms)

    def emissions(self, w, c, k, lengths):
        e = self.word(w)
        e = torch.where((w == UNK).unsqueeze(-1), self.unk_vec.expand_as(e), e)
        feats = [e]
        if self.args.char:
            B, T, L = c.shape
            ce = self.char(c.view(B * T, L)).transpose(1, 2)
            ce = F.relu(self.char_cnn(ce)).max(2)[0].view(B, T, -1)
            feats.append(ce)
        if self.args.case:
            feats.append(self.case(k))
        x = self.drop(torch.cat(feats, -1))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=w.shape[1])
        em = self.out(self.proj(h))
        return em if self.args.crf else F.log_softmax(em, -1)

    def decode_params(self):
        if self.args.crf:
            return self.crf.params()
        z = torch.zeros(self.mask_start.shape, device=self.mask_start.device)
        return self.mask_trans, self.mask_start, z

    def loss(self, w, c, k, y, lengths):
        em = self.emissions(w, c, k, lengths)
        mask = (torch.arange(w.shape[1], device=w.device)[None] < lengths.to(w.device)[:, None]).float()
        if self.args.crf:
            return self.crf.nll(em, y, mask)
        return F.nll_loss(em.reshape(-1, em.shape[-1]), y.reshape(-1), reduction="none").mul(mask.reshape(-1)).sum() / w.shape[0]


# ----------------------------------------------------------------------------
# Train / predict
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(models, items, bs=128, device="cpu"):
    """Ensemble: average emission scores and transition parameters, then Viterbi."""
    preds = []
    for m in models:
        m.eval()
    for _, chunk in batches(items, bs, False):
        w, c, k, _, lengths = collate(chunk, device)
        em = sum(m.emissions(w, c, k, lengths) for m in models) / len(models)
        ps = [m.decode_params() for m in models]
        tr, st, en = (sum(p[i] for p in ps) / len(ps) for i in range(3))
        preds.extend(viterbi(em, lengths, tr, st, en))
    return preds


def dev_eval(models, data, device):
    pred = predict(models, data.dev_t, device=device)
    pred_tags = [[data.id2tag[i] for i in p] for p in pred]
    return evaluate([s[1] for s in data.dev], pred_tags)


def train_one(args, data, seed, device, verbose=True):
    set_seed(seed)
    model = Tagger(args, data).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, bad, history = -1, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for _, chunk in batches(data.train_t, args.batch_size, True):
            w, c, k, y, lengths = collate(chunk, device)
            if args.word_dropout > 0:
                drop = (torch.rand_like(w, dtype=torch.float) < args.word_dropout) & (w != PAD)
                w = w.masked_fill(drop, UNK)
            loss = model.loss(w, c, k, y, lengths)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item()
        res = dev_eval([model], data, device)
        history.append({"epoch": ep, "train_loss": tot / (len(data.train_t) / args.batch_size), "dev_f1": res["f1"]})
        if res["f1"] > best:
            best, bad = res["f1"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose:
            print(f"  seed {seed} ep {ep:2d} loss {history[-1]['train_loss']:.3f} dev P {res['p']:.3f} "
                  f"R {res['r']:.3f} F1 {res['f1']:.4f} (best {best:.4f}) [{time.time() - t0:.1f}s]", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best, history


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def write_result(data, models, path, device):
    """Write result.txt line by line, mirroring test.txt (blank lines preserved)."""
    pred = predict(models, data.test_t, device=device)
    si, ti, n = 0, 0, 0
    with open(os.path.join(data.data_dir, "test.txt"), encoding="utf-8") as fin, \
            open(path, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                if ti > 0:
                    si, ti = si + 1, 0
                fout.write("\n")
                continue
            tok = line.split("\t")[0] if "\t" in line else line.split()[0]
            assert data.test[si][0][ti] == tok
            fout.write(f"{tok}\t{data.id2tag[pred[si][ti]]}\n")
            ti, n = ti + 1, n + 1
    assert si + (1 if ti else 0) == len(data.test)
    return n


def save_plot(histories, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, h in enumerate(histories):
        ax[0].plot([e["epoch"] for e in h], [e["train_loss"] for e in h], label=f"seed {i}")
        ax[1].plot([e["epoch"] for e in h], [e["dev_f1"] for e in h], label=f"seed {i}")
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].set(title="Dev entity-level F1", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    data = Data(args)
    data.data_dir = args.data_dir
    print(f"train {len(data.train)} / dev {len(data.dev)} / test {len(data.test)} sentences; tags={data.id2tag}")

    if args.mode == "quick":
        args.epochs = 5
        model, best, hist = train_one(args, data, args.seed, device)
        print("best dev F1", best)

    elif args.mode == "ablation":
        configs = [
            ("RNN (uni) + GloVe, softmax", dict(encoder="rnn", bidirectional=0, char=0, case=0, crf=0)),
            ("GRU (uni) + GloVe, softmax", dict(encoder="gru", bidirectional=0, char=0, case=0, crf=0)),
            ("LSTM (uni) + GloVe, softmax", dict(encoder="lstm", bidirectional=0, char=0, case=0, crf=0)),
            ("BiRNN + GloVe, softmax", dict(encoder="rnn", char=0, case=0, crf=0)),
            ("BiGRU + GloVe, softmax", dict(encoder="gru", char=0, case=0, crf=0)),
            ("BiLSTM + GloVe, softmax", dict(encoder="lstm", char=0, case=0, crf=0)),
            ("BiLSTM + GloVe + case, softmax", dict(encoder="lstm", char=0, case=1, crf=0)),
            ("BiLSTM + GloVe + char, softmax", dict(encoder="lstm", char=1, case=0, crf=0)),
            ("BiLSTM + GloVe + char + case, softmax", dict(encoder="lstm", char=1, case=1, crf=0)),
            ("BiGRU + GloVe + char + case + CRF", dict(encoder="gru", char=1, case=1, crf=1)),
            ("BiLSTM + GloVe + char + case + CRF", dict(encoder="lstm", char=1, case=1, crf=1)),
            ("BiLSTM + char + case + CRF, fine-tuned GloVe", dict(encoder="lstm", char=1, case=1, crf=1, finetune=1)),
            ("2-layer BiLSTM + char + case + CRF", dict(encoder="lstm", layers=2, char=1, case=1, crf=1)),
        ]
        rows = []
        for name, over in configs:
            a = copy.copy(args)
            for k_, v in over.items():
                setattr(a, k_, v)
            f1s = []
            for s in range(args.n_seeds):
                _, best, _ = train_one(a, data, args.seed + s, device, verbose=False)
                f1s.append(best)
                print(f"[{name}] seed {s}: dev F1 {best:.4f}", flush=True)
            rows.append({"config": name, "mean_f1": np.mean(f1s), "std_f1": np.std(f1s), "runs": json.dumps(f1s)})
            print(f"==> {name}: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}", flush=True)
            with open(os.path.join(args.out_dir, "ablation.csv"), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)

    else:  # final
        models, histories, bests = [], [], []
        for i in range(args.n_models):
            m, best, h = train_one(args, data, args.seed + i, device)
            models.append(m)
            histories.append(h)
            bests.append(best)
            print(f"model {i}: best dev F1 {best:.4f}", flush=True)
        single = float(np.mean(bests))
        res = dev_eval(models, data, device)
        print(f"\nSingle-model mean dev F1: {single:.4f}")
        print(f"Ensemble ({len(models)}) dev  P {res['p']:.4f} R {res['r']:.4f} F1 {res['f1']:.4f}")
        for t, v in res["per_type"].items():
            print(f"  {t:12s} P {v['p']:.3f} R {v['r']:.3f} F1 {v['f1']:.3f} support {v['support']}")
        with open(os.path.join(args.out_dir, "dev_results.json"), "w") as f:
            json.dump({"single_model_f1": bests, "ensemble": res, "args": vars(args), "histories": histories}, f, indent=1)
        save_plot(histories, os.path.join(args.out_dir, "training_curves.png"))
        n = write_result(data, models, os.path.join(args.out_dir, "result.txt"), device)
        print(f"Wrote {n} predicted tokens to {os.path.join(args.out_dir, 'result.txt')}")


if __name__ == "__main__":
    main()
