"""
xlmr_lid.py
===========

Fine-tunes a multilingual transformer (default: MuRIL) for token-level
Language Identification (LID) on code-mixed English-Hindi-Gujarati /
English-Hindi-Bengali data, and produces prediction files in the exact
CoNLL format required by the shared task:

    submission_<team_name>.zip
    ├── eng-hin-guj.conll
    └── eng-hin-ben.conll

Each .conll file: one token + TAB + label per line, sentences separated by
a single blank line. Labels: HIN, BEN, GUJ, ENG, UNI.

--------------------------------------------------------------------------
WHAT CHANGED VS. THE FIRST VERSION (and why)
--------------------------------------------------------------------------
1. Default model switched from xlm-roberta-base to google/muril-base-cased.
   MuRIL is pretrained on transliterated + native-script Indian languages
   and tends to outperform generic XLM-R on romanized code-mixed text.
2. Class-weighted loss (--class_weighting, on by default). Your label
   distribution is heavily imbalanced (e.g. HIN had only ~30 examples in
   the Bengali dev file vs ~4900 ENG). Because the task is scored with
   MACRO F1, rare classes matter just as much as frequent ones, so the
   loss now upweights rare labels so the model doesn't ignore them.
3. K-fold cross-validation (--kfold, default 5). With only ~480 sentences
   per language pair, a single 90/10 split gives a noisy, unreliable
   estimate of Macro F1. K-fold trains K models on different splits and
   reports an averaged Macro F1, which is a much more trustworthy signal
   of what's actually working. All K fold models are saved and can be
   ensembled at prediction time.
4. LR warmup + cosine schedule, which stabilizes fine-tuning on small
   datasets.
5. Prediction supports model ensembling: pass a comma-separated list of
   model directories to --model_dir and logits are averaged across all
   of them before taking argmax, which typically adds a few more points
   of Macro F1 over any single fold model.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------

1) K-fold cross-validated training (recommended) on a labeled CoNLL file:

   python xlmr_lid.py train \
       --train_file dev_eng_hin_guj.txt \
       --output_dir ./xlmr-lid-guj \
       --kfold 5 \
       --epochs 8

   This trains 5 models (one per fold), saved to
   ./xlmr-lid-guj/fold_0 ... ./xlmr-lid-guj/fold_4, prints per-fold and
   averaged Macro F1, and also trains one more model on ALL the data
   (saved to ./xlmr-lid-guj/final) for convenience.

   To just do a plain single train/val split like before, pass --kfold 1.

2) Predict on a test file, optionally ensembling multiple fold models:

   python xlmr_lid.py predict \
       --model_dir "./xlmr-lid-guj/fold_0,./xlmr-lid-guj/fold_1,./xlmr-lid-guj/fold_2,./xlmr-lid-guj/fold_3,./xlmr-lid-guj/fold_4" \
       --test_file test_eng_hin_guj.txt \
       --output_file eng-hin-guj.conll

   Or with a single model:

   python xlmr_lid.py predict \ 
       --model_dir ./xlmr-lid-guj/final \
       --test_file test_eng_hin_guj.txt \
       --output_file eng-hin-guj.conll

3) Repeat separately for the Bengali dataset, then zip:

   zip submission_<team_name>.zip eng-hin-guj.conll eng-hin-ben.conll

--------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------
- Only the FIRST subword of every token receives a label during training;
  the remaining subwords are masked with -100 so the loss/metrics are
  computed at the token level, matching the shared task's evaluation.
- At prediction time, the label assigned to a token is the label predicted
  for its first subword.
- Macro F1 (matching the task's official metric) is reported on held-out
  data during training via sklearn.

Install dependencies first:
    pip install transformers datasets seqeval scikit-learn torch --upgrade
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Data loading / writing
# --------------------------------------------------------------------------

def read_conll(path: str) -> List[List[Tuple[str, str]]]:
    """Reads a CoNLL file: token<TAB>label per line, blank line = sentence
    boundary. Returns a list of sentences, each a list of (token, label)
    tuples. If a line has no label column (tokens-only file), label is
    filled with a placeholder 'O'.
    """
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                token, label = parts[0], parts[1]
            else:
                token, label = parts[0], "O"
            current.append((token, label))
    if current:
        sentences.append(current)
    return sentences


def write_conll(sentences: List[List[Tuple[str, str]]], path: str) -> None:
    """Writes sentences (list of (token, label) tuples) to CoNLL format:
    token<TAB>label, blank line between sentences, exactly matching the
    format required by the shared task.
    """
    with open(path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(sentences):
            for token, label in sent:
                f.write(f"{token}\t{label}\n")
            if i != len(sentences) - 1:
                f.write("\n")


# --------------------------------------------------------------------------
# Shared training helpers
# --------------------------------------------------------------------------

def compute_class_weights(train_sents, label2id, cap=5.0):
    """Inverse-frequency class weights, capped to avoid a single very rare
    class exploding the loss / destabilizing training.
    """
    import torch

    counts = {lab: 0 for lab in label2id}
    for sent in train_sents:
        for _, lab in sent:
            counts[lab] += 1
    total = sum(counts.values())
    n_classes = len(label2id)

    weights = [0.0] * n_classes
    for lab, idx in label2id.items():
        c = max(counts[lab], 1)
        w = total / (n_classes * c)
        weights[idx] = min(w, cap)
    print("Class counts:", counts)
    print("Class weights:", {lab: round(weights[idx], 3) for lab, idx in label2id.items()})
    return torch.tensor(weights, dtype=torch.float)


def tokenize_and_align_builder(tokenizer, max_length):
    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)  # only first subword gets a label
                prev_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align


def make_compute_metrics(id2label, report_holder=None):
    """report_holder: optional dict that will be mutated in-place with the
    latest classification_report text and per-class f1 dict, so the caller
    can write it to disk after evaluation (the Trainer API only returns
    scalar metrics, so we stash the richer report here as a side channel).
    """
    from sklearn.metrics import f1_score, classification_report

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_preds, true_labels = [], []
        for pred_row, lab_row in zip(predictions, labels):
            for p, l in zip(pred_row, lab_row):
                if l != -100:
                    true_preds.append(id2label[p])
                    true_labels.append(id2label[l])

        macro_f1 = f1_score(true_labels, true_preds, average="macro", zero_division=0)
        micro_f1 = f1_score(true_labels, true_preds, average="micro", zero_division=0)
        report_text = classification_report(true_labels, true_preds, zero_division=0)
        report_dict = classification_report(true_labels, true_preds, zero_division=0, output_dict=True)
        print(report_text)
        if report_holder is not None:
            report_holder["text"] = report_text
            report_holder["dict"] = report_dict
        return {"macro_f1": macro_f1, "micro_f1": micro_f1}

    return compute_metrics


def build_weighted_trainer_class():
    """Returns a Trainer subclass with class-weighted CrossEntropyLoss.
    Built lazily inside a function so `transformers`/`torch` are only
    imported when actually training.
    """
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.label_smoothing = label_smoothing

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
            else:
                weight = None
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=weight, ignore_index=-100, label_smoothing=self.label_smoothing
            )
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def train_one_model(train_sents, val_sents, label2id, id2label, args, save_dir, use_class_weights=True):
    """Trains a single model on train_sents, evaluates on val_sents (if
    non-empty), saves to save_dir. Returns the eval metrics dict (or None
    if val_sents is empty).
    """
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
    )

    def to_hf_dict(sents):
        return {
            "tokens": [[t for t, _ in s] for s in sents],
            "ner_tags": [[label2id[l] for _, l in s] for s in sents],
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_and_align = tokenize_and_align_builder(tokenizer, args.max_length)

    train_ds = Dataset.from_dict(to_hf_dict(train_sents))
    train_ds = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)

    has_val = len(val_sents) > 0
    if has_val:
        val_ds = Dataset.from_dict(to_hf_dict(val_sents))
        val_ds = val_ds.map(tokenize_and_align, batched=True, remove_columns=val_ds.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    class_weights = compute_class_weights(train_sents, label2id, cap=args.class_weight_cap) if use_class_weights else None
    report_holder = {}

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch" if has_val else "no",
        save_total_limit=2,
        load_best_model_at_end=has_val,
        metric_for_best_model="macro_f1" if has_val else None,
        greater_is_better=True if has_val else None,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )

    WeightedTrainer = build_weighted_trainer_class()
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label, report_holder) if has_val else None,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )
    try:
        # transformers >= 5.x renamed this argument
        trainer = WeightedTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # older transformers versions
        trainer = WeightedTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()

    metrics = None
    if has_val:
        metrics = trainer.evaluate()
        print(f"[{save_dir}] Final validation metrics:", metrics)

        os.makedirs(save_dir, exist_ok=True)
        import json
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"eval_metrics": metrics, "per_class": report_holder.get("dict")},
                f, indent=2, default=str,
            )
        if "text" in report_holder:
            with open(os.path.join(save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(report_holder["text"])
        print(f"Saved metrics.json and classification_report.txt to {save_dir}")

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model + tokenizer saved to {save_dir}")
    return metrics


# --------------------------------------------------------------------------
# Training entry point (single split or k-fold)
# --------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    sentences = read_conll(args.train_file)
    print(f"Loaded {len(sentences)} sentences from {args.train_file}")

    label_set = sorted({lab for sent in sentences for _, lab in sent})
    print("Label set:", label_set)
    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}

    idxs = list(range(len(sentences)))
    random.shuffle(idxs)

    if args.kfold <= 1:
        # Original behavior: single train/val split.
        n_val = max(1, int(len(idxs) * args.val_ratio))
        val_idxs = set(idxs[:n_val])
        train_sents = [sentences[i] for i in idxs if i not in val_idxs]
        val_sents = [sentences[i] for i in idxs if i in val_idxs]
        print(f"Train sentences: {len(train_sents)} | Val sentences: {len(val_sents)}")
        train_one_model(
            train_sents, val_sents, label2id, id2label, args,
            save_dir=args.output_dir, use_class_weights=args.class_weighting,
        )
        return

    # K-fold cross-validation
    k = args.kfold
    fold_size = len(idxs) // k
    fold_metrics = []
    for fold in range(k):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold != k - 1 else len(idxs)
        val_idx_set = set(idxs[val_start:val_end])
        train_sents = [sentences[i] for i in idxs if i not in val_idx_set]
        val_sents = [sentences[i] for i in idxs if i in val_idx_set]

        print(f"\n===== Fold {fold + 1}/{k} | Train: {len(train_sents)} | Val: {len(val_sents)} =====")
        save_dir = os.path.join(args.output_dir, f"fold_{fold}")
        metrics = train_one_model(
            train_sents, val_sents, label2id, id2label, args,
            save_dir=save_dir, use_class_weights=args.class_weighting,
        )
        fold_metrics.append(metrics["eval_macro_f1"])

    print("\n===== K-fold results =====")
    for i, m in enumerate(fold_metrics):
        print(f"Fold {i}: Macro F1 = {m:.4f}")
    print(f"Average Macro F1 across {k} folds: {np.mean(fold_metrics):.4f} (+/- {np.std(fold_metrics):.4f})")

    if args.train_final_on_all:
        print("\n===== Training final model on ALL data (no held-out val) =====")
        final_dir = os.path.join(args.output_dir, "final")
        train_one_model(
            sentences, [], label2id, id2label, args,
            save_dir=final_dir, use_class_weights=args.class_weighting,
        )
        print(f"Final model (trained on 100% of data) saved to {final_dir}")
        print("Use this directory, or the fold_* directories ensembled together, for prediction.")


# --------------------------------------------------------------------------
# Prediction (supports single model or ensemble of multiple model dirs)
# --------------------------------------------------------------------------

def predict(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch.nn.functional as F

    model_dirs = [d.strip() for d in args.model_dir.split(",") if d.strip()]
    print(f"Using {len(model_dirs)} model(s) for prediction: {model_dirs}")

    sentences = read_conll(args.test_file)
    print(f"Loaded {len(sentences)} sentences from {args.test_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizers = []
    models = []
    for d in model_dirs:
        tok = AutoTokenizer.from_pretrained(d)
        mdl = AutoModelForTokenClassification.from_pretrained(d)
        mdl.eval()
        mdl.to(device)
        tokenizers.append(tok)
        models.append(mdl)

    id2label = models[0].config.id2label

    output_sentences = []
    with torch.no_grad():
        for sent in sentences:
            tokens = [t for t, _ in sent]
            if len(tokens) == 0:
                output_sentences.append([])
                continue

            avg_probs = None
            word_ids_ref = None
            for tok, mdl in zip(tokenizers, models):
                enc = tok(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = mdl(**enc).logits[0]
                probs = F.softmax(logits, dim=-1).cpu().numpy()

                word_ids = enc.word_ids(batch_index=0)
                if word_ids_ref is None:
                    word_ids_ref = word_ids

                # Aggregate first-subword probs per word position (aligned
                # across models since we re-tokenize with each model's own
                # tokenizer but align back to the same `tokens` list order).
                per_word_probs = {}
                seen = set()
                for pos, wid in enumerate(word_ids):
                    if wid is None or wid in seen:
                        continue
                    seen.add(wid)
                    per_word_probs[wid] = probs[pos]

                n_labels = probs.shape[-1]
                model_word_probs = np.zeros((len(tokens), n_labels), dtype=np.float64)
                for wid, p in per_word_probs.items():
                    model_word_probs[wid] = p

                if avg_probs is None:
                    avg_probs = model_word_probs
                else:
                    avg_probs += model_word_probs

            avg_probs /= len(models)
            pred_ids = np.argmax(avg_probs, axis=-1)
            token_labels = [id2label[int(p)] for p in pred_ids]

            # Fallback: tokens truncated by every model get UNI
            row_sums = avg_probs.sum(axis=-1)
            for i, s in enumerate(row_sums):
                if s == 0:
                    token_labels[i] = "UNI"

            output_sentences.append(list(zip(tokens, token_labels)))

    write_conll(output_sentences, args.output_file)
    print(f"Predictions written to {args.output_file}")

    # Sanity check: token counts must match exactly
    n_in = sum(len(s) for s in sentences)
    n_out = sum(len(s) for s in output_sentences)
    assert n_in == n_out, f"Token count mismatch! input={n_in} output={n_out}"
    assert len(sentences) == len(output_sentences), "Sentence count mismatch!"
    print("Sanity check passed: sentence and token counts match the input file.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(description="Multilingual token-level Language ID (code-mixed)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune a model on a labeled CoNLL file")
    p_train.add_argument("--train_file", required=True, help="Path to labeled CoNLL file (token<TAB>label)")
    p_train.add_argument("--output_dir", default="./xlmr-lid-model")
    p_train.add_argument("--model_name", default="google/muril-base-cased",
                          help="e.g. google/muril-base-cased, ai4bharat/indic-bert, xlm-roberta-base, xlm-roberta-large")
    p_train.add_argument("--epochs", type=int, default=5,
                          help="With only ~480 sentences, more than 5-6 epochs risks overfitting to the dev "
                               "set even though held-out CV macro F1 looks good; watch for a growing gap "
                               "between dev-CV score and real test-set score as epochs increase.")
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-5)
    p_train.add_argument("--warmup_ratio", type=float, default=0.1)
    p_train.add_argument("--lr_scheduler_type", default="cosine")
    p_train.add_argument("--max_length", type=int, default=128)
    p_train.add_argument("--val_ratio", type=float, default=0.1, help="Used only when --kfold 1")
    p_train.add_argument("--kfold", type=int, default=5, help="Number of CV folds. Use 1 for a single train/val split.")
    p_train.add_argument("--train_final_on_all", action="store_true", default=True,
                          help="After k-fold CV, also train one more model on 100%% of the data (saved to output_dir/final)")
    p_train.add_argument("--no_train_final_on_all", dest="train_final_on_all", action="store_false")
    p_train.add_argument("--class_weighting", action="store_true", default=True,
                          help="Use inverse-frequency class-weighted loss (recommended given label imbalance)")
    p_train.add_argument("--no_class_weighting", dest="class_weighting", action="store_false")
    p_train.add_argument("--class_weight_cap", type=float, default=5.0,
                          help="Max multiplier for any single class's weight. Lower = gentler upweighting of "
                               "rare classes, reduces risk of overfitting to rare-class patterns in a small "
                               "dev set at the cost of real-world (unseen test) precision on common classes.")
    p_train.add_argument("--label_smoothing", type=float, default=0.05,
                          help="Label smoothing factor (0-1). Small values (0.05-0.1) reduce overconfidence "
                               "and typically help generalization on small datasets.")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="Predict labels on a test CoNLL/token file")
    p_pred.add_argument("--model_dir", required=True,
                         help="Path to a directory saved by `train`, or a comma-separated list of "
                              "directories to ensemble (e.g. fold_0,fold_1,fold_2,fold_3,fold_4)")
    p_pred.add_argument("--test_file", required=True, help="Test file: tokens, one per line, blank line per sentence")
    p_pred.add_argument("--output_file", required=True, help="Where to write predictions, e.g. eng-hin-guj.conll")
    p_pred.add_argument("--max_length", type=int, default=128)
    p_pred.set_defaults(func=predict)

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)