import os
import json
import sys
import numpy as np
import torch
import random
import copy
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPDependencyParser as dep_parser
from nltk.tokenize import sent_tokenize, word_tokenize
from csv_reader import *

PAD = 0
BOS = 1
EOS = 2
UNK = 3
MASK = 4
TITLE = 5
MAX_LENGTH = 100


class Vocab:
    def __init__(self, vocab_file, content_file, vocab_size=50000):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4, '_TITLE_': 5}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]', '_TITLE_']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1, '_TITLE_': 1}
        if not os.path.exists(vocab_file):
            self.build_vocab(content_file, vocab_file)
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0

    @staticmethod
    def build_vocab(corpus_file, vocab_file):
        word2count = {}
        for line in open(corpus_file):
            words = line.strip().split()
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
        word2count = list(word2count.items())
        word2count.sort(key=lambda k: k[1], reverse=True)
        write = open(vocab_file, 'w')
        for word_pair in word2count:
            write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
        write.close()

    def load_vocab(self, vocab_file, vocab_size=None, min_count=5):
        for line in open(vocab_file):
            term_ = line.strip().split('\t')
            if len(term_) != 2:
                continue
            word, count = term_
            if int(count) <= min_count:
                break
            assert word not in self._word2id
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._wordcount[word] = int(count)
            if vocab_size > 0 and len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False):
        result = [self.word2id(word) for word in sent]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result


class Example:
    """
    Each example is one data pair
        src: title (has oov)
        tgt: comment (oov has extend ids if use_oov else has oov)
        memory: tag (oov has extend ids)
    """

    def __init__(self, content, title, label, vocab, use_depparse=False, use_neighbor=False, use_hierarchical=False):
        if use_depparse:
            title_words = [word[0] for word in title]
            self.ori_title = ' '.join(title_words)
            content_words = []
            for sent in content:
                content_words.extend([word[0] for word in sent])
            self.ori_content = ' '.join(content_words)
        else:
            self.ori_title = title
            title_words = word_tokenize(title)
            self.ori_content = content
            content_words = word_tokenize(content)
        self.title = vocab.sent2id(title_words)
        if not use_hierarchical:
            self.content = self.title + vocab.sent2id(content_words)
            self.length_for_sort = len(self.content)
        self.label = label
        if use_neighbor and not use_hierarchical:
            if use_depparse:
                flat_content = []
                for s in content:
                    flat_content.extend(s)
                self.content_neighbor_index = self.get_neighbor_index(flat_content, True)
            else:
                self.content_neighbor_index = self.get_neighbor_index(self.content)
        if use_hierarchical:
            if use_depparse:
                self.sentence_content = [self.title] + [vocab.sent2id([word[0] for word in sentence]) for sentence in
                                                        content]
            else:
                self.sentence_content = sent_tokenize(self.ori_content)
                self.sentence_content = [self.title] + [vocab.sent2id(word_tokenize(sentence)) for sentence in
                                                        self.sentence_content]
            self.length_for_sort = len(self.sentence_content)
            self.sentence_content_max_len = min(max([len(c) for c in self.sentence_content]), MAX_LENGTH)
            self.sentence_content, self.sentence_content_mask = Batch.padding(self.sentence_content,
                                                                              self.sentence_content_max_len,
                                                                              limit_length=True)
            if use_neighbor:
                if use_depparse:
                    self.sentence_neighbor_index = [self.get_neighbor_index(sentence, True) for sentence in
                                                    content]
                else:
                    self.sentence_neighbor_index = [self.get_neighbor_index(sentence) for sentence in
                                                    self.sentence_content]
                self.sentence_neighbor_index, self.sentence_neighbor_mask = Batch.neighbor_index_padding(
                    self.sentence_neighbor_index, True)
                self.neighbor_index = self.get_neighbor_index(self.sentence_content)
                # self.neighbor_mask = self.get_neighbor_mask(self.sentence_content)

    @staticmethod
    def get_neighbor_mask(sentence, use_depparse=False):
        # here len(sentence)+1 is to insert the root node
        mask = [[0 for _ in range(len(sentence) + 1)] for _ in range(len(sentence))]
        for i in range(len(sentence)):
            # note i+1 is the real i, because of the inserted root node
            if i > 0:
                mask[i][i] = 1
            if i < len(sentence) - 1:
                mask[i][i + 2] = 1
            if use_depparse:
                if int(sentence[i][1]) == 0:
                    mask[i][0] = 1
                else:
                    mask[i][int(sentence[i][1]) + i + 1] = 1
                    mask[int(sentence[i][1]) + i][i + 1] = 1
        return mask

    @staticmethod
    def get_neighbor_index(sentence, use_depparse=False):
        indices = [[] for _ in range(len(sentence))]
        for i in range(len(sentence)):
            # assume 0 index is padding, note i+1 is the real i
            if i > 0:
                indices[i].append(i)
            if i < len(sentence) - 1:
                indices[i].append(i + 2)
            if use_depparse:
                indices[i].append((int(sentence[i][1]) + i + 1))
                indices[int(sentence[i][1]) + i].append(i + 1)
        # in case there is only one sentence in the document and neighbor index is empty
        if len(sentence) == 1:
            indices[0].append(0)
        return indices

    def bow(self, content, maxlen=MAX_LENGTH):
        bow = {}
        for word_id in content:
            if word_id not in bow:
                bow[word_id] = 0
            bow[word_id] += 1
        bow = list(bow.items())
        bow.sort(key=lambda k: k[1], reverse=True)
        bow.insert(0, (UNK, 1))
        return [word_id[0] for word_id in bow[:maxlen]]


class Batch:
    """
    Each batch is a mini-batch of data

    """

    def __init__(self, example_list, model):
        max_len = MAX_LENGTH
        self.model = model
        self.examples = example_list
        self.label = np.array([e.label for e in example_list], dtype=np.long)
        self.batch_size = len(example_list)
        if model == 'h_attention':
            self.sentence_content = [np.array(e.sentence_content, dtype=np.long) for e in example_list]
            self.sentence_content_mask = [np.array(e.sentence_content_mask, dtype=np.int32) for e in example_list]
            self.sentence_content_len = [len(e.sentence_content) for e in example_list]
            max_sent_num = max(self.sentence_content_len)
            # sentence level mask
            self.sentence_mask, _ = self.padding([[1 for _ in range(d)] for d in self.sentence_content_len],
                                                 max_sent_num, limit_length=False)
        elif model == 'slstm':
            content_max_len = max([len(e.content) for e in example_list])
            self.content, self.content_mask = self.padding([e.content for e in example_list], content_max_len,
                                                           limit_length=True)
        elif model == 'glstm':
            content_max_len = max([len(e.content) for e in example_list])
            self.content, self.content_mask = self.padding([e.content for e in example_list], content_max_len,
                                                           limit_length=True)
            self.neighbor_index, self.neighbor_mask = self.neighbor_index_padding(
                [e.content_neighbor_index for e in example_list], limit_length=True)
        elif model == 'hglstm':
            self.sentence_content = [np.array(e.sentence_content, dtype=np.long) for e in example_list]
            self.sentence_content_mask = [np.array(e.sentence_content_mask, dtype=np.int32) for e in example_list]
            self.sentence_content_len = [len(e.sentence_content) for e in example_list]
            max_sent_num = max(self.sentence_content_len)
            # sentence level mask
            self.sentence_mask, _ = self.padding([[1 for _ in range(d)] for d in self.sentence_content_len],
                                                 max_sent_num, limit_length=False)
            self.sentence_neighbor_index = [np.array(e.sentence_neighbor_index, dtype=np.int32) for e in example_list]
            self.sentence_neighbor_mask = [np.array(e.sentence_neighbor_mask, dtype=np.int32) for e in example_list]
            self.neighbor_index, self.neighbor_mask = self.neighbor_index_padding(
                [e.neighbor_index for e in example_list], limit_length=False)
            # self.neighbor_mask = self.neighbor_mask_padding([e.neighbor_mask for e in example_list], max_sent_num, limit_length=False)

        self.to_tensor()

    def get_length(self, examples, max_len):
        length = []
        for e in examples:
            if len(e) > max_len:
                length.append(max_len)
            else:
                length.append(len(e))
        assert len(length) == len(examples)
        return length

    def to_tensor(self):
        if self.model == 'h_attention':
            self.sentence_content = [torch.from_numpy(src) for src in self.sentence_content]
            self.sentence_content_mask = [torch.from_numpy(mask) for mask in self.sentence_content_mask]
            self.sentence_content_len = torch.from_numpy(np.array(self.sentence_content_len, dtype=np.long))
            self.sentence_mask = torch.from_numpy(np.array(self.sentence_mask, dtype=np.int32))
        elif self.model == 'slstm':
            self.content = torch.from_numpy(np.array(self.content, dtype=np.long))
            self.content_mask = torch.from_numpy(np.array(self.content_mask, dtype=np.int))
        elif self.model == 'glstm':
            self.content = torch.from_numpy(np.array(self.content, dtype=np.long))
            self.content_mask = torch.from_numpy(np.array(self.content_mask, dtype=np.int))
            self.neighbor_mask = torch.from_numpy(
                np.array([np.array(m, dtype=np.int) for m in self.neighbor_mask], dtype=np.int))
            self.neighbor_index = torch.from_numpy(
                np.array([np.array(m, dtype=np.long) for m in self.neighbor_index], dtype=np.long))
        elif self.model == 'hglstm':
            self.sentence_content = [torch.from_numpy(src) for src in self.sentence_content]
            self.sentence_content_mask = [torch.from_numpy(mask) for mask in self.sentence_content_mask]
            self.sentence_content_len = torch.from_numpy(np.array(self.sentence_content_len, dtype=np.long))
            self.sentence_mask = torch.from_numpy(np.array(self.sentence_mask, dtype=np.int32))
            self.sentence_neighbor_index = [torch.from_numpy(index) for index in self.sentence_neighbor_index]
            self.sentence_neighbor_mask = [torch.from_numpy(mask) for mask in self.sentence_neighbor_mask]
            self.neighbor_mask = torch.from_numpy(
                np.array([np.array(m, dtype=np.int) for m in self.neighbor_mask], dtype=np.int))
            self.neighbor_index = torch.from_numpy(
                np.array([np.array(m, dtype=np.long) for m in self.neighbor_index], dtype=np.long))

        self.label = torch.from_numpy(self.label)

    @staticmethod
    def padding(batch, max_len, limit_length=True):
        if limit_length:
            max_len = min(max_len, MAX_LENGTH)
        result = []
        mask_batch = []
        for s in batch:
            l = copy.deepcopy(s)
            m = [1. for _ in range(len(l))]
            l = l[:max_len]
            m = m[:max_len]
            while len(l) < max_len:
                l.append(0)
                m.append(0.)
            result.append(l)
            mask_batch.append(m)
        return result, mask_batch

    @staticmethod
    def neighbor_index_padding(batch, limit_length=True):
        # sentence, word, neighbor
        max_neighbor = max([max([len(w) for w in s]) for s in batch])
        max_word = max([len(s) for s in batch])
        if limit_length:
            max_word = min(max_word, MAX_LENGTH)
        result = []
        mask = []
        for s in batch:
            l = copy.deepcopy(s)
            l = [t[:max_neighbor] for t in l[:max_word]]
            for t in l[:max_word]:
                for i in range(len(t)):
                    # neighbor index exceeds max word number
                    if t[i] > max_word:
                        t[i] = 0
            m = [[1 for n in w] for w in l]
            for i, w in enumerate(l):
                while len(w) < max_neighbor:
                    w.append(0)
                    m[i].append(0)
            while len(l) < max_word:
                l.append([0 for _ in range(max_neighbor)])
                m.append([0 for _ in range(max_neighbor)])
            assert len(l) == len(m)
            result.append(l)
            mask.append(m)
        return result, mask


class DataLoader:
    def __init__(self, config, task, has_dev, batch_size, vocab, model, use_iterator, use_depparse=False,
                 no_train=False, debug=False):
        assert MAX_LENGTH == config.max_sentence_len, (MAX_LENGTH, config.max_sentence_len)
        self.debug = debug
        self.vocab = vocab
        self.batch_size = batch_size
        self.model = model
        if not no_train:
            if use_depparse:
                self.train_data = self.read_json(task, 'train_preprocess_data.json')
            else:
                self.train_data = self.read_data(task, 'train.csv')
            if has_dev:
                if use_depparse:
                    self.dev_data = self.read_json(task, 'dev_preprocess_data.json')
                else:
                    self.dev_data = self.read_data(task, 'dev.csv')
            else:
                self.split_dev(config.dev_split)
            self.dev_batches = self.make_batch(self.dev_data, batch_size, model=model)
            self.train_batches = self.make_batch(self.train_data, batch_size, model=model)
            random.shuffle(self.train_batches)
        if use_depparse:
            self.test_data = self.read_json(task, 'test_preprocess_data.json')
        else:
            self.test_data = self.read_data(task, 'test.csv')
        self.test_batches = self.make_batch(self.test_data, batch_size, model=model)

    def split_dev(self, rate=0.1):
        self.dev_data = self.train_data[:round(len(self.train_data) * rate)]
        self.train_data = self.train_data[round(len(self.train_data) * rate):]

    class ExampleIterator:
        def __init__(self, data, task, model):
            self.data = data
            self.task = task
            self.model = model
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.data):
                self.index = 0
            if self.task == 'ag' or self.task == 'amazon':
                label, title, content = self.data[self.index]
                self.index += 1
                if self.model == 'hglstm':
                    return Example(content, title, int(label) - 1, self.vocab, use_neighbor=True, use_hierarchical=True)
                elif self.model == 'h_attention':
                    return Example(content, title, int(label) - 1, self.vocab, use_hierarchical=True)
                elif self.model == 'glstm':
                    return Example(content, title, int(label) - 1, self.vocab, use_neighbor=True)
                elif self.model == 'slstm':
                    return Example(content, title, int(label) - 1, self.vocab)
                else:
                    print('ERROR!')
                    return

    class BatchIterator:
        def __init__(self, example_iter, batch_size, model):
            self.example_iter = example_iter
            self.batch_size = batch_size
            self.model = model

        def __iter__(self):
            return self

        def __next__(self):
            examples = []
            for _ in range(self.batch_size):
                examples.append(next(self.example_iter))
            return Batch(examples, self.model)

    def read_data(self, data_file, fname):
        result = []
        if data_file == 'ag':
            data = read_ag(fname)
            for d in data:
                label, title, content = d
                if self.model == 'hglstm':
                    result.append(Example(content, title, int(label) - 1, self.vocab, use_neighbor=True,
                                          use_hierarchical=True))
                elif self.model == 'h_attention':
                    result.append(Example(content, title, int(label) - 1, self.vocab, use_hierarchical=True))
                elif self.model == 'glstm':
                    result.append(Example(content, title, int(label) - 1, self.vocab, use_neighbor=True))
                elif self.model == 'slstm':
                    result.append(Example(content, title, int(label) - 1, self.vocab))
                else:
                    print('ERROR!')
        elif data_file == 'amazon':
            data = read_amazon(fname)
            print('finished reading data', flush=True)
            for d in tqdm(data):
                label, title, content = d
                if self.model == 'hglstm':
                    result.append(Example(content, title, int(label) - 1, self.vocab, use_neighbor=True,
                                          use_hierarchical=True))
                elif self.model == 'h_attention':
                    result.append(Example(content, title, int(label) - 1, self.vocab, use_hierarchical=True))
                elif self.model == 'glstm':
                    result.append(Example(content, title, int(label) - 1, self.vocab, use_neighbor=True))
                elif self.model == 'slstm':
                    result.append(Example(content, title, int(label) - 1, self.vocab))
                else:
                    print('ERROR!')
        return result

    def read_json(self, data_file, fname):
        result = []
        if data_file == 'ag':
            data = json.load(open(os.path.join('./data/ag_news', fname)))
            for d in data:
                label, title, content = d
                if self.model == 'hglstm':
                    result.append(
                        Example(content, title, int(label), self.vocab, use_hierarchical=True, use_depparse=True,
                                use_neighbor=True))
                elif self.model == 'h_attention':
                    result.append(
                        Example(content, title, int(label), self.vocab, use_hierarchical=True, use_depparse=True))
                elif self.model == 'glstm':
                    result.append(
                        Example(content, title, int(label), self.vocab, use_neighbor=True, use_depparse=True))
                elif self.model == 'slstm':
                    result.append(Example(content, title, int(label), self.vocab, use_depparse=True))
        elif data_file == 'amazon':
            data = read_amazon(fname)
            for d in data:
                label, title, content = d
                if self.model in ['h_attention', 'hglstm']:
                    result.append(Example(content, title, int(label) - 1, self.vocab, True))
                else:
                    result.append(Example(content, title, int(label) - 1, self.vocab))
        return result

    def make_batch(self, data, batch_size, model):
        batches = []
        data.sort(key=lambda d: d.length_for_sort)
        for i in range(0, len(data), batch_size):
            batches.append(Batch(data[i:i + batch_size], model))
        return batches


if __name__ == '__main__':
    pass
