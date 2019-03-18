import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
from torch.nn.utils.rnn import pad_sequence
from Data import *

import numpy as np


class attentive_pooling(nn.Module):
    def __init__(self, hidden_size):
        super(attentive_pooling, self).__init__()
        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, mask=None):
        h = torch.tanh(self.w(memory))
        score = torch.squeeze(self.u(h), -1)
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, 1)
        return s


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirec):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout, bidirectional=bidirec,
                           batch_first=True)

    def forward(self, input, lengths):
        length, indices = torch.sort(lengths, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = list(torch.unbind(length, dim=0))
        embs = pack(torch.index_select(input, dim=0, index=indices), input_length, batch_first=True)
        outputs, _ = self.rnn(embs)
        outputs = unpack(outputs, batch_first=True)[0]
        outputs = torch.index_select(outputs, dim=0, index=ind)
        return outputs


class hierarchical_attention(nn.Module):
    def __init__(self, config, vocab, use_cuda, pretrain=None):
        super(hierarchical_attention, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.word_encoder = Encoder(config.emb_size, config.emb_size, 1, config.dropout, config.bidirec)
        self.word_attentive_pool = attentive_pooling(config.emb_size * 2)
        self.sentence_encoder = Encoder(config.emb_size * 2, config.emb_size * 2, 1,
                                        config.dropout, config.bidirec)
        self.sentence_attentive_pool = attentive_pooling(config.encoder_hidden_size*2)
        self.w_context = nn.Linear(config.encoder_hidden_size * 4, config.encoder_hidden_size*2, bias=False)
        self.w_out = nn.Linear(config.encoder_hidden_size*2, config.label_size)
        self.tanh = nn.Tanh()
        self.config = config
        self.log_softmax = nn.LogSoftmax()

    def encode(self, contents, contents_mask, contents_length, sent_mask):
        sent_vec_batch = []
        for content, content_mask, content_length in zip(contents, contents_mask, contents_length):
            length = torch.sum(content_mask, -1)
            emb = self.embedding(content)
            context = self.word_encoder(emb, length)
            sent_vec = self.word_attentive_pool(context, content_mask)
            sent_vec_batch.append(sent_vec)
            assert len(sent_vec) == content_length, (len(sent_vec), content_length)  # sentence number
        sent_vec_batch = pad_sequence(sent_vec_batch, batch_first=True)
        sent_hidden = self.sentence_encoder(sent_vec_batch, contents_length)
        sent_hidden = self.tanh(self.w_context(sent_hidden))
        state = self.sentence_attentive_pool(sent_hidden, sent_mask)
        return sent_hidden, state

    def forward(self, batch, use_cuda):
        src, src_mask, src_len = batch.sentence_content, batch.sentence_content_mask, batch.sentence_content_len
        sent_mask = batch.sentence_mask
        if use_cuda:
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            src_len = src_len.cuda()
            sent_mask = sent_mask.cuda()
        context, state = self.encode(src, src_mask, src_len, sent_mask)
        output = self.w_out(state)
        return output

