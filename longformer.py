from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix
import json, evaluate
import numpy as np
import os
import pandas as pd
import random
TRAIN = 23
TEST = 6

random.seed(9)

def load_dataset(path):
    data = {'train': {}, 'test': {}}
    eval = pd.read_csv('data/VivesEval/VivesDebate_eval.csv')
    data['train']['label'] = []
    data['train']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []
    n_tr = 0
    lst = os.listdir(path)
    random.shuffle(lst)

    for file in lst:
        eval_df = eval.loc[eval['DEBATE'] == file.split('.')[0]]
        f = open(path+file, "r")
        spl = f.read()

        try:
            winner = eval.iloc[eval_df['SCORE'].idxmax(), :]

            if winner['STANCE'] == 'Favour':
                if n_tr < TRAIN:
                    data['train']['label'].append(0)
                    data['train']['text'].append(spl)
                    n_tr += 1
                else:
                    data['test']['label'].append(0)
                    data['test']['text'].append(spl)
            else:
                if n_tr < TRAIN:
                    data['train']['label'].append(1)
                    data['train']['text'].append(spl)
                    n_tr += 1
                else:
                    data['test']['label'].append(1)
                    data['test']['text'].append(spl)
        except:
            if n_tr < TRAIN:
                data['train']['label'].append(1)
                data['train']['text'].append(spl)
                n_tr += 1
            else:
                data['test']['label'].append(1)
                data['test']['text'].append(spl)

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def tokenize_sequence(samples):
    if 'text2' in samples.keys():
        return tknz(samples["text"], samples["text2"], padding="max_length", truncation=True)
    else:
        return tknz(samples["text"], padding=True, truncation=True)


def load_model(n_lb):
    tokenizer_hf = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = AutoModelForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=n_lb, ignore_mismatched_sizes=True)

    return tokenizer_hf, model


def load_local_model(path):
    tokenizer_hf = AutoTokenizer.from_pretrained('markussagen/xlm-roberta-longformer-base-4096')
    model = AutoModelForSequenceClassification.from_pretrained(path)

    return tokenizer_hf, model


def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


def train_model(mdl, tknz, data):

    training_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True
    )

    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        tokenizer=tknz,
        data_collator=DataCollatorWithPadding(tokenizer=tknz),
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer


if __name__ == "__main__":

    PRETRAIN = True

    num_labels = 2

    path_data = 'data/VivesDebate_NL/'

    # LOAD DATA FOR THE MODE
    dataset = load_dataset(path_data)

    if PRETRAIN:

        # LOAD PRE_TRAINED Longformer
        tknz, mdl = load_model(num_labels)

        # TOKENIZE THE DATA
        tokenized_data = dataset.map(tokenize_sequence, batched=True)

        # TRAIN THE MODEL
        trainer = train_model(mdl, tknz, tokenized_data)

        # GENERATE PREDICTIONS FOR DEV AND TEST
        test_predictions = trainer.predict(tokenized_data['test'])
        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='macro')

        print('Macro F1 score TEST:', mf1_test)
        print('Confusion matrix')
        print(confusion_matrix(tokenized_data['test']['label'], test_predict))

    else:

        path_model = ''

        tknz, mdl = load_local_model(path_model)

        shuffled_dataset = dataset.shuffle(seed=42)

        tokenized_data = shuffled_dataset.map(tokenize_sequence, batched=True)

        trainer = Trainer(mdl)

        test_predictions = trainer.predict(tokenized_data['test'])
        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        #print(dev_predict)

        #print(test_predict)

        mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='macro')

        print('Macro F1 score TEST:', mf1_test)
        print('Confusion matrix')
        print(confusion_matrix(tokenized_data['test']['label'], test_predict))