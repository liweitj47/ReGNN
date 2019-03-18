from csv_reader import *
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser as dep_parser

parser = dep_parser()


def parse_position(word, word_id):
    word, pos, parent_id, label = word.split('\t')
    parent_id = int(parent_id)
    if parent_id == 0:
        return word, 0
    return word, parent_id - (word_id + 1)


def parse_sentence(sentence):
    parse_sentence = next(parser.raw_parse(sentence)).to_conll(4).split('\n')
    words = [parse_position(word, idx) for idx, word in enumerate(parse_sentence) if
             len(word) != 0 and idx != len(parse_sentence) - 1]
    return words


def parse_document(doc):
    sents = sent_tokenize(doc)
    parse_sents = parser.raw_parse_sents(sents)
    words = [
        [parse_position(word, idx) for idx, word in enumerate(next(sent).to_conll(4).split('\n')) if len(word) != 0] for
        sent in parse_sents]
    return words


def csv_to_json(data_file, fname):
    result = []
    if data_file == 'ag':
        data = read_ag(fname)
        for d in data:
            label, title, content = d
            label = int(label) - 1
            title = parse_sentence(title)
            content = parse_document(content)
            result.append((label, title, content))
    json.dump(result, open(fname.split('.')[0] + '_preprocess_data.json', 'w'), indent=4, ensure_ascii=False)
    return result


if __name__ == '__main__':
    csv_to_json('ag', 'train.csv')
