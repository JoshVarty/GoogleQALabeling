import os
import gc
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from math import floor, ceil
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, RandomSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import (BertPreTrainedModel, BertTokenizer, BertConfig,
                          BertModel, get_cosine_schedule_with_warmup
                          )

if os.getcwd().endswith('src'):
    # We want to be working in the root directory
    os.chdir('../')


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class PipeLineConfig:
    def __init__(self, lr, warmup, accum_steps, epochs, seed, expname, head_tail, freeze, question_weight,
                 answer_weight, fold, train):
        self.lr = lr
        self.warmup = warmup
        self.accum_steps = accum_steps
        self.epochs = epochs
        self.seed = seed
        self.expname = expname
        self.head_tail = head_tail
        self.freeze = freeze
        self.question_weight = question_weight
        self.answer_weight = answer_weight
        self.fold = fold
        self.train = train


# From the Ref Kernel's
def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    # TODO This seems wrong?
    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(title, question, answer, max_sequence_length=290, t_max_len=30, q_max_len=128, a_max_len=128):
    # 350+128+30 = 508 + 4 = 512

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" % (
            max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))
        q_len_head = round(q_new_len / 2)
        q_len_tail = -1 * (q_new_len - q_len_head)
        a_len_head = round(a_new_len / 2)
        a_len_tail = -1 * (a_new_len - a_len_head)  ## Head+Tail method .
        t = t[:t_new_len]
        if config.head_tail:
            q = q[:q_len_head] + q[q_len_tail:]
            a = a[:a_len_head] + a[a_len_tail:]
        else:
            q = q[:q_new_len]
            a = a[:a_new_len]  ## No Head+Tail ,usual processing

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    # stoken = ["[CLS]"] + title  + question  + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def _get_stoken_output(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    return stoken


def compute_input_tokens(df, columns, tokenizer, max_sequence_length):
    input_tokens, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        tokens = _get_stoken_output(t, q, a, tokenizer, max_sequence_length)
        input_tokens.append(tokens)
    return input_tokens


def compute_input_arays(df, columns, tokenizer, max_sequence_length, t_max_len=30, q_max_len=128, a_max_len=128):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length, t_max_len, q_max_len, a_max_len)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):

        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):

        input_ids = self.inputs[0][idx]
        input_masks = self.inputs[1][idx]
        input_segments = self.inputs[2][idx]
        lengths = self.lengths[idx]
        if self.labels is not None:  # targets
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])


## Stolen from transformer code base without any noble intention.
class CustomBert(BertPreTrainedModel):

    def __init__(self, config):
        super(CustomBert, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bn = nn.BatchNorm1d(1024)
        self.linear = nn.Linear(config.hidden_size, 1024)
        self.classifier = nn.Linear(1024, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        lin_output = F.relu(self.bn(self.linear(
            pooled_output)))  ## Note : This Linear layer is added without expert supervision . This will worsen the results .
        ## But you are smarter than me , so you will figure out,how to customize better.
        lin_output = self.dropout(lin_output)
        logits = self.classifier(lin_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def train_model(train_loader, optimizer, criterion, scheduler, config):
    model.train()
    avg_loss = 0.
    avg_loss_1 = 0.
    avg_loss_2 = 0.
    avg_loss_3 = 0.
    avg_loss_4 = 0.
    avg_loss_5 = 0.
    # tk0 = tqdm(enumerate(train_loader),total =len(train_loader))
    optimizer.zero_grad()
    for idx, batch in enumerate(train_loader):

        input_ids, input_masks, input_segments, labels, _ = batch
        input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(
            device), input_segments.to(device), labels.to(device)

        output_train = model(input_ids=input_ids.long(),
                             labels=None,
                             attention_mask=input_masks,
                             token_type_ids=input_segments,
                             )
        logits = output_train[0]  # output preds
        loss1 = criterion(logits[:, 0:9], labels[:, 0:9])
        loss2 = criterion(logits[:, 9:10], labels[:, 9:10])
        loss3 = criterion(logits[:, 10:21], labels[:, 10:21])
        loss4 = criterion(logits[:, 21:26], labels[:, 21:26])
        loss5 = criterion(logits[:, 26:30], labels[:, 26:30])
        loss = config.question_weight * loss1 + config.answer_weight * loss2 + config.question_weight * loss3 + config.answer_weight * loss4 + config.question_weight * loss5
        # loss =(config.question_weight*criterion(logits[:,0:21], labels[:,0:21]) + config.answer_weight*criterion(logits[:,21:30], labels[:,21:30]))/config.accum_steps
        loss.backward()
        if (i + 1) % config.accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss += loss.item() / (len(train_loader) * config.accum_steps)
        avg_loss_1 += loss1.item() / (len(train_loader) * config.accum_steps)
        avg_loss_2 += loss2.item() / (len(train_loader) * config.accum_steps)
        avg_loss_3 += loss3.item() / (len(train_loader) * config.accum_steps)
        avg_loss_4 += loss4.item() / (len(train_loader) * config.accum_steps)
        avg_loss_5 += loss5.item() / (len(train_loader) * config.accum_steps)
        del input_ids, input_masks, input_segments, labels

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss, avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, avg_loss_5


def val_model(val_loader, val_shape, batch_size=8):
    avg_val_loss = 0.
    model.eval()  # eval mode

    valid_preds = np.zeros((val_shape, len(target_cols)))
    original = np.zeros((val_shape, len(target_cols)))

    # tk0 = tqdm(enumerate(val_loader))
    with torch.no_grad():

        for idx, batch in enumerate(val_loader):
            input_ids, input_masks, input_segments, labels, _ = batch
            input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(
                device), input_segments.to(device), labels.to(device)

            output_val = model(input_ids=input_ids.long(),
                               labels=None,
                               attention_mask=input_masks,
                               token_type_ids=input_segments,
                               )
            logits = output_val[0]  # output preds

            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx * batch_size: (idx + 1) * batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx * batch_size: (idx + 1) * batch_size] = labels.detach().cpu().squeeze().numpy()

        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

        # np.save("preds.npy", preds)
        # np.save("actuals.npy", original)

        rho_val = np.mean([spearmanr(original[:, i], preds[:, i]).correlation for i in range(preds.shape[1])])
        print('\r val_spearman-rho: %s' % (str(round(rho_val, 5))), end=100 * ' ' + '\n')

        for i in range(len(target_cols)):
            print(i, spearmanr(original[:, i], preds[:, i]))
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
    return avg_val_loss, score / len(target_cols)


def predict_result(model, test_loader, batch_size=32):
    test_preds = np.zeros((len(test), len(target_cols)))

    model.eval()
    tk0 = tqdm(enumerate(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(input_ids=x_batch[0].to(device),
                            labels=None,
                            attention_mask=x_batch[1].to(device),
                            token_type_ids=x_batch[2].to(device),
                            )
            predictions = outputs[0]
            test_preds[idx * batch_size: (idx + 1) * batch_size] = predictions.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    ## load the data
    data = Path('data')
    train = pd.read_csv(data / 'train.csv')
    #TODO: Remove, we're just doing this to speed things up
    train = train.iloc[:999]
    test = pd.read_csv(data / 'test.csv')
    submission = pd.read_csv(data / 'sample_submission.csv')

    target_cols = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
                   'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
                   'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
                   'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
                   'question_type_compare', 'question_type_consequence', 'question_type_definition',
                   'question_type_entity',
                   'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation',
                   'question_type_spelling', 'question_well_written', 'answer_helpful', 'answer_level_of_information',
                   'answer_plausible', 'answer_relevance', 'answer_satisfaction', 'answer_type_instructions',
                   'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']
    input_categories = list(train.columns[[1, 2, 5]])

    config = PipeLineConfig(lr=3e-5,
                            warmup=0.05,
                            accum_steps=4,
                            epochs=1,
                            seed=42,
                            expname='uncased_1',
                            head_tail=True,
                            freeze=False,
                            question_weight=0.7,
                            answer_weight=0.3,
                            fold=3,
                            train=True)

    tokenizer = BertTokenizer.from_pretrained("data/bert-base-uncased-vocab.txt", do_lower_case=True)
    bert_model_config = 'data/bert-base-uncased/bert_config.json'
    bert_config = BertConfig.from_json_file(bert_model_config)
    bert_config.num_labels = len(target_cols)

    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_model_file = 'bert_pytorch.bin'


    NUM_FOLDS = config.fold  # change this
    SEED = config.seed
    BATCH_SIZE = 8
    epochs = config.epochs  # change this
    ACCUM_STEPS = config.accum_steps
    MAX_SEQ_LEN = 512

    test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=MAX_SEQ_LEN, t_max_len=30, q_max_len=239, a_max_len=239)
    lengths_test = np.argmax(test_inputs[0] == 0, axis=1)       # Find the first zero entry in each example
    lengths_test[lengths_test == 0] = test_inputs[0].shape[1]   # For every entry without zero, store the full length (512)

    print(do_lower_case, bert_model, device, output_model_file)
    print(test_inputs)

    test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    result = np.zeros((len(test), len(target_cols)))

    kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED)

    # test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
    # test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
    # result = np.zeros((len(test), len(target_cols)))

    y_train = train[target_cols].values  # dummy

    print(bcolors.FAIL, f"For Every Fold, Train {epochs} Epochs", bcolors.ENDC)
    if config.train:
        for fold, (train_index, val_index) in enumerate(kf.split(train.values, y_train)):

            print(bcolors.HEADER, "Current Fold:", fold, bcolors.ENDC)

            train_df, val_df = train.iloc[train_index], train.iloc[val_index]
            print("Train and Valid Shapes are", train_df.shape, val_df.shape)

            print(bcolors.HEADER, "Preparing train datasets....", bcolors.ENDC)

            inputs_train = compute_input_arays(train_df, input_categories, tokenizer, max_sequence_length=MAX_SEQ_LEN, t_max_len=30, q_max_len=239, a_max_len=239)
            outputs_train = compute_output_arrays(train_df, columns=target_cols)
            outputs_train = torch.tensor(outputs_train, dtype=torch.float32)
            lengths_train = np.argmax(inputs_train[0] == 0, axis=1)
            lengths_train[lengths_train == 0] = inputs_train[0].shape[1]

            print(bcolors.HEADER, "Preparing Valid datasets....", bcolors.ENDC)

            inputs_valid = compute_input_arays(val_df, input_categories, tokenizer, max_sequence_length=MAX_SEQ_LEN, t_max_len=30, q_max_len=239, a_max_len=239)
            outputs_valid = compute_output_arrays(val_df, columns=target_cols)
            outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
            lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
            lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]

            print(bcolors.HEADER, "Preparing Dataloaders Datasets....", bcolors.ENDC)

            train_set = QuestDataset(inputs=inputs_train, lengths=lengths_train, labels=outputs_train)
            train_sampler = RandomSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)

            valid_set = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
            valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

            model = CustomBert.from_pretrained('data/bert-base-uncased/', config=bert_config)
            model.zero_grad()
            model.to(device)
            torch.cuda.empty_cache()
            if config.freeze:  ## This is basically using out of the box bert model while training only the classifier head with our data .
                for param in model.bert.parameters():
                    param.requires_grad = False
            model.train()

            i = 0
            best_avg_loss = 100.0
            best_score = -1.
            best_param_loss = None
            best_param_score = None
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.8},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr, eps=4e-5)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=4e-5)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                        num_training_steps=epochs * len(train_loader) // ACCUM_STEPS)
            print("Training....")

            for epoch in range(config.epochs):

                torch.cuda.empty_cache()

                start_time = time.time()
                avg_loss, avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, avg_loss_5 = train_model(train_loader,
                                                                                                   optimizer, criterion,
                                                                                                   scheduler, config)
                avg_val_loss, score = val_model(valid_loader, val_shape=val_df.shape[0])
                elapsed_time = time.time() - start_time

                print(bcolors.OKGREEN,
                      'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t train_loss={:.4f} \t train_loss_1={:.4f} \t train_loss_2={:.4f} \t train_loss_3={:.4f} \t train_loss_4={:.4f}  \t train_loss_5={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
                          epoch + 1, epochs, avg_loss, avg_val_loss, avg_loss, avg_loss_1, avg_loss_2, avg_loss_3,
                          avg_loss_4, avg_loss_5, score, elapsed_time),
                      bcolors.ENDC
                      )

                if best_avg_loss > avg_val_loss:
                    i = 0
                    best_avg_loss = avg_val_loss
                    best_param_loss = model.state_dict()

                if best_score < score:
                    best_score = score
                    best_param_score = model.state_dict()
                    print('best_param_score_{}_{}.pt'.format(config.expname, fold + 1))
                    torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname, fold + 1))
                else:
                    i += 1

            model.load_state_dict(best_param_score)
            result += predict_result(model, test_loader, BATCH_SIZE)
            print('best_param_score_{}_{}.pt'.format(config.expname, fold + 1))
            torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname, fold + 1))

            result /= NUM_FOLDS

        del train_df, val_df, model, optimizer, criterion, scheduler
        torch.cuda.empty_cache()
        del valid_loader, train_loader, valid_set, train_set
        torch.cuda.empty_cache()
        gc.collect()
