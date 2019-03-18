import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.utils.rnn import pad_sequence


def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # clone linear for 4 times, query, key, value, output
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            assert mask.dim() == 4  # batch, head, seq_len, seq_len
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_size):
        super(Attentive_Pooling, self).__init__()
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


class Neighbor_Attn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Neighbor_Attn, self).__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.u = nn.Linear(hidden_size, 1)
        self.V = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, g, neighbor_index, neighbor_mask):
        '''
        :param x: (batch, sent, emb)
        :param h: (batch, sent, hidden)
        :param g: (batch, hidden)
        :param neighbor_index: (batch, sent, neighbor)
        :param neighbor_mask: (batch, sent, neighbor+1)
        :return:
        '''
        shape = neighbor_index.size()
        new_h = torch.cat([torch.unsqueeze(torch.zeros_like(g), 1), h], 1)
        neighbor_index = neighbor_index.view(shape[0], -1)
        # new_h = torch.gather(new_h, 1, neighbor_index).view(shape[0], shape[1], shape[2], -1)
        ind = torch.unsqueeze(torch.arange(shape[0]), 1).repeat(1, shape[1] * shape[2])
        new_h = new_h[ind.long(), neighbor_index.long()].view(shape[0], shape[1], shape[2], -1)
        # new_h = torch.cat([torch.unsqueeze(g, 1), h], 1)
        # big_h = new_h.repeat([1, shape[1], 1]).view([shape[0], shape[1], shape[1] + 1, shape[2]])
        # neighbors = self.Wn(big_h) * torch.unsqueeze(neighbor_mask.float(), -1)
        neighbors = self.Wn(new_h) * torch.unsqueeze(neighbor_mask.float(), -1)
        # batch, sent, sent, hidden
        s = torch.unsqueeze(self.Wh(h) + self.U(x) + torch.unsqueeze(self.V(g), 1), 2) + neighbors
        score = F.softmax(torch.squeeze(self.u(s), -1) * (1 - neighbor_mask).float() * 1e-25, -1)
        hn = torch.sum(torch.unsqueeze(score, -1) * neighbors, 2)
        return hn


class SCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wh = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 4)
        self.neighbor_attn = Neighbor_Attn(input_size, hidden_size)

    def forward(self, x, h, c, g, neighbor_index, neighbor_mask):
        hn = self.neighbor_attn(x, h, g, neighbor_index, neighbor_mask)
        gates = self.Wh(h) + self.U(x) + self.Wn(hn) + torch.unsqueeze(self.V(g), 1)
        i, f, o, u = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        return new_h, new_c


class GCell(nn.Module):
    def __init__(self, hidden_size):
        super(GCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 2)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.attn_pooling = Attentive_Pooling(hidden_size)

    def forward(self, g, c_g, h, c, mask):
        ''' assume dim=1 is word'''
        # this can use attentive pooling
        # h_avg = torch.mean(h, 1)
        h_avg = self.attn_pooling(h)
        f, o = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg)), self.hidden_size, dim=-1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), 1) + self.u(h)) - torch.unsqueeze((1 - mask) * 1e-25, -1).float()
        f_w = F.softmax(f_w, 1)
        new_c = f * c_g + torch.sum(c * f_w, 1)
        new_g = o * torch.tanh(new_c)
        return new_g, new_c


class GLSTM(nn.Module):
    def __init__(self, config, vocab, use_cuda, pretrain=None):
        super(GLSTM, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.hidden_size = config.encoder_hidden_size
        self.s_cell = SCell(config.emb_size, config.encoder_hidden_size)
        self.g_cell = GCell(config.encoder_hidden_size)
        self.w_out = nn.Linear(config.encoder_hidden_size, config.label_size)
        self.num_layers = config.num_layers

    def forward(self, batch, use_cuda):
        word = batch.content
        word_mask = batch.content_mask
        neighbor_index, neighbor_mask = batch.neighbor_index, batch.neighbor_mask
        if use_cuda:
            word = word.cuda()
            word_mask = word_mask.cuda()
            neighbor_mask = neighbor_mask.cuda()
            neighbor_index = neighbor_index.cuda()
        word_emb = self.embedding(word)
        init_h_states = self.embedding(word) * torch.unsqueeze(word_mask.float(), -1)
        init_c_states = self.embedding(word) * torch.unsqueeze(word_mask.float(), -1)
        init_g = torch.mean(init_h_states, 1)
        init_c_g = torch.mean(init_c_states, 1)
        for l in range(self.num_layers):
            init_h_states = init_h_states * torch.unsqueeze(word_mask.float(), -1)
            init_c_states = init_c_states * torch.unsqueeze(word_mask.float(), -1)
            new_h_states, new_c_states = self.s_cell(word_emb, init_h_states, init_c_states, init_g, neighbor_index,
                                                     neighbor_mask)
            new_g, new_c_g = self.g_cell(init_g, init_c_g, init_h_states, init_c_states, word_mask)
            init_h_states, init_c_states = new_h_states, new_c_states
            init_g, init_c_g = new_g, new_c_g
        return self.w_out(new_g)


class HGLSTM(nn.Module):
    def __init__(self, config, vocab, use_cuda, pretrain=None):
        super(HGLSTM, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.hidden_size = config.encoder_hidden_size
        self.s_cell = SCell(config.emb_size, config.encoder_hidden_size)
        self.g_cell = GCell(config.encoder_hidden_size)
        self.hs_cell = SCell(config.encoder_hidden_size, config.encoder_hidden_size)
        self.hg_cell = GCell(config.encoder_hidden_size)
        self.w_out = nn.Linear(config.encoder_hidden_size, config.label_size)
        self.num_layers = config.num_layers

    @staticmethod
    def get_hidden_before(hidden_states):
        shape = hidden_states.size()
        start = torch.zeros(shape[0], 1, shape[2]).to(hidden_states.device)
        return torch.cat([start, hidden_states[:, :-1, :]], 1)

    @staticmethod
    def get_hidden_after(hidden_states):
        shape = hidden_states.size()
        end = torch.zeros(shape[0], 1, shape[2]).to(hidden_states.device)
        return torch.cat([hidden_states[:, 1:, :], end], 1)

    def encode(self, contents, contents_mask, contents_neighbor_index, contents_neighbor_mask):
        sent_h_batch = []
        sent_c_batch = []
        for content, content_mask, content_neighbor_index, content_neighbor_mask in zip(contents, contents_mask,
                                                                                        contents_neighbor_index,
                                                                                        contents_neighbor_mask):
            word_emb = self.embedding(content)
            init_h_states = self.embedding(content) * torch.unsqueeze(content_mask.float(), -1)
            init_c_states = self.embedding(content) * torch.unsqueeze(content_mask.float(), -1)
            init_g = torch.mean(init_h_states, 1)
            init_c_g = torch.mean(init_c_states, 1)
            for l in range(self.num_layers):
                init_h_states = init_h_states * torch.unsqueeze(content_mask.float(), -1)
                init_c_states = init_c_states * torch.unsqueeze(content_mask.float(), -1)
                new_h_states, new_c_states = self.s_cell(word_emb, init_h_states, init_c_states, init_g,
                                                         content_neighbor_index,
                                                         content_neighbor_mask)
                new_g, new_c_g = self.g_cell(init_g, init_c_g, init_h_states, init_c_states, content_mask)
                init_h_states, init_c_states = new_h_states, new_c_states
                init_g, init_c_g = new_g, new_c_g
            sent_h_batch.append(new_g)
            sent_c_batch.append(new_c_g)
        sent_h_batch = pad_sequence(sent_h_batch, batch_first=True)
        sent_c_batch = pad_sequence(sent_c_batch, batch_first=True)
        return sent_h_batch, sent_c_batch

    def forward(self, batch, use_cuda):
        src, src_mask = batch.sentence_content, batch.sentence_content_mask
        sent_mask = batch.sentence_mask
        sentence_neighbor_index, sentence_neighbor_mask = batch.sentence_neighbor_index, batch.sentence_neighbor_mask
        neighbor_index, neighbor_mask = batch.neighbor_index, batch.neighbor_mask
        if use_cuda:
            src = [s.cuda() for s in src]
            src_mask = [s.cuda() for s in src_mask]
            sentence_neighbor_mask = [s.cuda() for s in sentence_neighbor_mask]
            sent_mask = sent_mask.cuda()
            neighbor_mask = neighbor_mask.cuda()
            neighbor_index = neighbor_index.cuda()
        init_h_states, init_c_states = self.encode(src, src_mask, sentence_neighbor_index, sentence_neighbor_mask)
        sent_emb = init_h_states
        init_g = torch.mean(init_h_states, 1)
        init_c_g = torch.mean(init_c_states, 1)
        for l in range(self.num_layers):
            init_h_states = init_h_states * torch.unsqueeze(sent_mask.float(), -1)
            init_c_states = init_c_states * torch.unsqueeze(sent_mask.float(), -1)
            new_h_states, new_c_states = self.s_cell(sent_emb, init_h_states, init_c_states, init_g, neighbor_index,
                                                     neighbor_mask)
            new_g, new_c_g = self.g_cell(init_g, init_c_g, init_h_states, init_c_states, sent_mask)
            init_h_states, init_c_states = new_h_states, new_c_states
            init_g, init_c_g = new_g, new_c_g
        return self.w_out(new_g)
