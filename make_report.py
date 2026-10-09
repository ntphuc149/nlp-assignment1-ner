"""Builds 115522605_report.pdf from output/{ablation.csv,dv_results.json,training_curves.png}.
Requires reportlab (report tooling only; not needed to train the model)."""
import csv
import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
SID = "115522605"

res = json.load(open(os.path.join(OUT, "dv_results.json")))
args = res["args"]
abl = list(csv.DictReader(open(os.path.join(OUT, "ablation.csv"))))
ens = res["ensemble"]
single = res["single_model_f1"]
stopped = [len(h) for h in res["histories"]]
best_ep = [max(h, key=lambda e: e["dev_f1"])["epoch"] for h in res["histories"]]

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.5, leading=10.5, spaceAfter=0)


def P(t, s=B):
    return Paragraph(t, s)


def table(rows, widths, hl=None, font=8.5):
    rows = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e9ef")),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if hl is not None:
        st.append(("BACKGROUND", (0, hl), (-1, hl), colors.HexColor("#fff2cc")))
    t.setStyle(TableStyle(st))
    return t


S = []
S.append(P("Assignment 1 - Named Entity Recognition on WNUT-2016", ParagraphStyle("T", parent=ss["Title"], fontSize=17)))
S.append(P(f"Student ID: {SID} &nbsp;|&nbsp; Natural Language Processing, 2026", ParagraphStyle("S", parent=B, alignment=1)))

S.append(P("1. Task and approach", H1))
S.append(P(
    "The task is token-level sequence labelling of tweets with 10 fine-grained entity types (person, geo-loc, company, "
    "facility, product, musicartist, movie, sportsteam, tvshow, other) using the BIO scheme (21 tags). Systems are scored by "
    "exact-match entity-level precision / recall / F1 (CoNLL-2003 protocol). I implemented, in PyTorch, a recurrent tagger: "
    "pre-trained GloVe-Twitter word embeddings, a character-level CNN and a casing / token-type feature feed a "
    "bidirectional LSTM, and a linear-chain CRF produces the tag sequence. The CRF layer, the BIO-constrained Viterbi decoder "
    "and the entity-level scorer are written from scratch (no external NER or CRF library); the scorer follows "
    "<i>conlleval</i> semantics."))

S.append(P("2. Data and preprocessing", H1))
S.append(table([
    ["Split", "Sentences", "Tokens", "Notes"],
    ["train", "2,394", "46,469", "1,496 entities; 94.7% of tokens are O"],
    ["dev", "1,000", "16,261", "used for model selection / early stopping"],
    ["test", "3,850", "61,908", "no labels; predictions written to result.txt"]],
    [2.2 * cm, 2.4 * cm, 2.2 * cm, 9.5 * cm]))
S.append(Spacer(1, 4))
S.append(P(
    "The class distribution is highly skewed (e.g. 449 B-person vs. 34 B-tvshow / 34 B-movie in train), which explains the "
    "large per-class differences reported below."))
S.append(P("<b>Word normalisation (for embedding lookup).</b> GloVe-Twitter (glove.twitter.27B, 200-d) was trained on "
           "lower-cased, normalised tweets, so each token is mapped to an ordered list of candidate keys and the first one found is used: "
           "@mentions -&gt; &lt;user&gt;; URLs -&gt; &lt;url&gt;; numbers -&gt; &lt;number&gt;; hashtags -&gt; the word without '#'; "
           "otherwise the lower-cased token, then the token with letter elongations squeezed (\"sooooo\" -&gt; \"soo\" -&gt; \"so\"). "
           "Tokens with no match map to a single &lt;unk&gt; vector which is learned during training; during training 5% of words are "
           "randomly replaced by &lt;unk&gt; (word dropout) so that this vector is useful for unseen words. The vocabulary contains only "
           "words that exist in GloVe (word lists of train/dev/test are used only to decide which GloVe rows to load; no labels of "
           "dev/test are used)."))
S.append(P("<b>Character input.</b> Each token (truncated to 20 characters, original casing preserved) is a sequence of character ids; "
           "this lets the model exploit capitalisation, punctuation, '@'/'#' prefixes and sub-word clues for out-of-vocabulary names."))
S.append(P("<b>Casing / token-type feature.</b> A 10-way category (lower, UPPER, Title, mIxed, digit, @user, url, #hashtag, punct, other) "
           "embedded to 16 dimensions. This restores the case information lost by lower-casing for the embedding lookup."))
S.append(P("<b>Batching.</b> Sentences are padded per batch (batch size 32) and packed before the RNN so that padding never affects the states."))

S.append(P("3. Model architecture", H1))
S.append(table([
    ["Component", "Configuration"],
    ["Word embedding", "GloVe-Twitter 200-d, <b>frozen</b>; one trainable &lt;unk&gt; vector"],
    ["Character encoder", "char embedding 30-d -> Conv1D (50 filters, kernel 3) -> ReLU -> max-pool over characters"],
    ["Casing feature", "10 token types -> 16-d embedding"],
    ["Input", "concat(200 + 50 + 16 = 266-d) -> dropout 0.5"],
    ["Encoder", "1-layer bidirectional LSTM, hidden 256 per direction (GRU / vanilla RNN / more layers are options)"],
    ["Projection", "Linear(512 -> 256) -> tanh -> dropout 0.5 -> Linear(256 -> 21) = emission scores"],
    ["Output layer", "linear-chain CRF (learned transition / start / end scores). Illegal BIO moves "
                     "(O -> I-x, B-x -> I-y, start -> I-x) are masked with -1e4 in training and decoding"],
    ["Decoding", "Viterbi; ensemble = average of the emission scores and of the CRF transition parameters of 5 models"]],
    [3.6 * cm, 13.5 * cm]))
S.append(Spacer(1, 4))
S.append(P("Freezing GloVe is deliberate: with only 2.4k training sentences, fine-tuning the 200-d vectors over-fits (see ablation)."))

S.append(P("4. Training process", H1))
S.append(P(
    f"Loss: CRF negative log-likelihood (per sentence). Optimiser: Adam (lr {args['lr']}, weight decay {args['weight_decay']}), "
    f"gradient clipping at 5, batch size {args['batch_size']}, dropout {args['dropout']}, up to {args['epochs']} epochs. After every epoch "
    f"the model is scored on dev (entity-level F1); the best-dev checkpoint is kept and training stops after {args['patience']} epochs without "
    f"improvement. The final system trains {args['n_models']} models with different seeds ({args['seed']}-{args['seed'] + args['n_models'] - 1}) and "
    f"ensembles them. The five runs stopped after {', '.join(map(str, stopped))} epochs (best epochs {', '.join(map(str, best_ep))}). "
    "Hyper-parameters were chosen by hand from the ablation below, based on dev F1 only; the test set was never scored."))
S.append(Image(os.path.join(OUT, "training_curves.png"), width=16 * cm, height=16 * cm * 3.5 / 10))
S.append(P("Figure 1. Training loss and dev F1 of the five final models. Dev F1 plateaus at about 0.42-0.45 after ~8-10 epochs "
           "while the training loss keeps decreasing, i.e. further training mostly over-fits; early stopping on dev F1 is therefore important.", CAP))

S.append(P("5. Results", H1))
S.append(P("5.1 Architecture ablation (dev set)", H2))
S.append(P("Every configuration was trained with 3 random seeds under the same optimiser and early-stopping protocol; "
           "the table gives the mean and standard deviation of the best dev F1."))
rows = [["Configuration", "Dev F1 (mean ± std)"]]
for r in abl:
    rows.append([r["config"], f"{float(r['mean_f1']):.4f} ± {float(r['std_f1']):.4f}"])
hl = 1 + [r["config"] for r in abl].index("BiLSTM + GloVe + char + case + CRF")
S.append(table(rows, [11.5 * cm, 5.6 * cm], hl=hl))
S.append(P("The highlighted row is the architecture used for the final system. The 2-layer variant has the highest mean "
           "(0.4645) but its gain over the 1-layer model (0.4512) is within one standard deviation (0.015), so I kept the simpler model.", CAP))

S.append(P("5.2 Final system (dev set)", H2))
S.append(P(f"Single models: best dev F1 = {', '.join(f'{x:.4f}' for x in single)} (mean {sum(single) / len(single):.4f}). "
           f"<b>5-model ensemble: precision {ens['p']:.4f}, recall {ens['r']:.4f}, F1 {ens['f1']:.4f}.</b> "
           "The ensemble is +2.3 F1 above the average single model. Dev scores are slightly optimistic because the "
           "checkpoint and hyper-parameters were selected on dev; the test F1 is expected to be lower."))
rows = [["Entity type", "Precision", "Recall", "F1", "Support (dev)"]]
for t, v in sorted(ens["per_type"].items(), key=lambda kv: -kv[1]["f1"]):
    rows.append([t, f"{v['p']:.3f}", f"{v['r']:.3f}", f"{v['f1']:.3f}", v["support"]])
rows.append(["<b>overall</b>", f"<b>{ens['p']:.3f}</b>", f"<b>{ens['r']:.3f}</b>", f"<b>{ens['f1']:.3f}</b>",
             sum(v["support"] for v in ens["per_type"].values())])
S.append(table(rows, [4 * cm, 3 * cm, 3 * cm, 3 * cm, 4 * cm]))

S.append(P("6. Observations", H1))
obs = [
    "<b>Gated units matter.</b> Vanilla RNNs (0.333 uni- / 0.331 bi-directional) are ~5 F1 below GRU (0.383 / 0.386) and LSTM (0.388 / 0.391); "
    "a likely cause is that they are harder to optimise and forget context quickly. LSTM is marginally better than GRU. Bi-directionality gives only +0.3 F1 for LSTM with "
    "word-only input, possibly because tweets are short, so most useful context is already available from the left.",
    "<b>Orthographic features give the largest gain.</b> Adding either casing features (+5.0) or the character CNN (+4.6) to a word-only BiLSTM raises F1 "
    "from 0.391 to ~0.44. In tweets, capitalisation, '@', '#' and spelling variants are strong entity cues that lower-cased GloVe cannot see. "
    "The two features are largely redundant (both together: 0.434 with softmax, within noise of either alone).",
    "<b>CRF helps and stabilises.</b> With char + case features, replacing softmax by CRF improves F1 from 0.434 to 0.451 and cuts the seed variance "
    "(0.020 -> 0.006), since transition scores and BIO constraints remove inconsistent tag sequences (e.g. I-person after O).",
    "<b>Do not fine-tune GloVe on this little data.</b> Fine-tuning drops F1 from 0.451 to 0.361: a plausible reason is that the vectors of words seen in training drift away "
    "from their neighbours while words only present in dev/test keep their original vectors, which breaks the shared embedding space (not verified).",
    "<b>Ensembling</b> reduces variance across seeds (single-model F1 ranges 0.433-0.467) and adds ~2 F1 points, at 5x training cost.",
    "<b>Per-type behaviour.</b> person (F1 0.72) and geo-loc (0.54) are the best types: frequent in train and with distinctive surface forms. The weakest types "
    "are musicartist (0.05, recall 0.02), tvshow (0.00, only 2 dev mentions), product (0.21) and other (0.25). These have few training examples and are "
    "semantically ambiguous with common words, so the model is conservative: precision (0.56) is much higher than recall (0.41) overall. The class often depends on world knowledge that 2.4k tweets cannot supply (a hypothesis; "
    "I did not run a systematic error analysis of individual sentences).",
    "<b>Limitations / possible improvements.</b> Contextual embeddings (ELMo/BERT-style), gazetteer features, class-weighted loss or oversampling of rare types, and training the "
    "final model on train+dev would probably improve recall on rare types; they were left out to stay within the required RNN/GloVe setting.",
]
for o in obs:
    S.append(P("• " + o))

S.append(P("7. Reproducibility", H1))
S.append(P(
    f"All code is in <font face='Courier'>{SID}.py</font>. <font face='Courier'>python {SID}.py --mode ablation</font> reproduces the ablation table and "
    "<font face='Courier'>python 115522605.py --mode final</font> trains the ensemble and writes result.txt (runs were done on a Kaggle GPU with seeds 42-46). "
    "result.txt has exactly the same line structure as test.txt (one 'token&lt;TAB&gt;tag' per line, blank lines between tweets, original order)."))

SimpleDocTemplate(os.path.join(OUT, f"{SID}_report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=f"{SID} NER report", author=SID).build(S)
print("ok")
