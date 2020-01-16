import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from math import floor, ceil
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel


if os.getcwd().endswith('src'):
    # When working locally, we want to be working in the root directory
    os.chdir('../')


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
            pooled_output)))  # Note : This Linear layer is added without expert supervision . This will worsen the results .
        # But you are smarter than me , so you will figure out,how to customize better.
        lin_output = self.dropout(lin_output)
        logits = self.classifier(lin_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


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
        a_len_tail = -1 * (a_new_len - a_len_head)  # Head+Tail method.
        t = t[:t_new_len]
        if config.head_tail:
            q = q[:q_len_head] + q[q_len_tail:]
            a = a[:a_len_head] + a[a_len_tail:]
        else:
            q = q[:q_new_len]
            a = a[:a_new_len]  # No Head+Tail ,usual processing

    return t, q, a


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


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    # stoken = ["[CLS]"] + title  + question  + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


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


if __name__ == '__main__':
    # load the data
    data = Path('data')
    train = pd.read_csv(data/'train.csv')
    test = pd.read_csv(data/'test.csv')
    submission = pd.read_csv(data/'sample_submission.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    MAX_SEQ_LEN = 512

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

    tokenizer = BertTokenizer.from_pretrained(data/"bert-base-uncased-vocab.txt", do_lower_case=True)

    test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=MAX_SEQ_LEN, t_max_len=30, q_max_len=239, a_max_len=239)
    lengths_test = np.argmax(test_inputs[0] == 0, axis=1)       # Find the first zero entry in each example
    lengths_test[lengths_test == 0] = test_inputs[0].shape[1]   # For every entry without zero, store the full length (512)

    checkpoints = ['best_param_score_uncased_1_1.pt', 'best_param_score_uncased_1_2.pt', 'best_param_score_uncased_1_3.pt']

    bert_model_config = data/'bert-base-uncased/bert_config.json'
    bert_config = BertConfig.from_json_file(bert_model_config)
    bert_config.num_labels = len(target_cols)

    BATCH_SIZE = 8
    test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    result = np.zeros((len(test), len(target_cols)))

    for checkpoint in checkpoints:

        # create model
        model = CustomBert(config=bert_config)
        model.to(device)

        # load saved weights
        checkpoint = torch.load(checkpoint)

        # initialize
        model.load_state_dict(checkpoint)

        # run on test data
        result += predict_result(model, test_loader, BATCH_SIZE)

    # Average of all models
    result = result / len(checkpoints)

    # save as submission
    submission.loc[:, 'question_asker_intent_understanding':] = result
    # TODO: Should we do this?
    # submission.loc[~submission['qa_id'].isin(qa_id_list),'question_type_spelling']=0.0
    # submission.loc[submission['qa_id'].isin(qa_id_list),'question_type_spelling'] = 1.0

    submission.to_csv('submission.csv', index=False)
    submission.head()
