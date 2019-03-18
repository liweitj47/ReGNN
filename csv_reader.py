import csv
import os
from sklearn.datasets import fetch_20newsgroups

csv.register_dialect('my_dialect', delimiter=',', quotechar='\"', doublequote=True)

data_dir = './data/'


def read_ag(fname):
    data_file = os.path.join(data_dir, 'ag_news', fname)
    data = []
    with open(data_file, 'r') as f:
        lines = csv.reader(f, 'my_dialect')
        for line in lines:
            label, title, content = line
            data.append((label, title, content))
    return data


def read_amazon(fname):
    data_file = os.path.join(data_dir, 'amazon', fname)
    data = []
    with open(data_file, 'r') as f:
        lines = csv.reader(f, 'my_dialect')
        for line in lines:
            label, title, content = line
            data.append((label, title, content))
    return data


def read_yelp(fname):
    data_file = os.path.join(data_dir, 'yelp', fname)
    data = []
    with open(data_file, 'r') as f:
        lines = csv.reader(f, 'my_dialect')
        for line in lines:
            label, content = line
            data.append((label, content))
    return data


def read_yahoo(fname):
    data_file = os.path.join(data_dir, 'yahoo', fname)
    data = []
    with open(data_file, 'r') as f:
        lines = csv.reader(f, 'my_dialect')
        for line in lines:
            label, q_title, q_content, answer = line
            data.append((label, q_title, q_content, answer))
    return data


def read_ohsumed(dname):
    data = []
    labels = os.listdir(dname)
    for label in labels:
        label_dir = os.path.join(dname, label)
        files = os.listdir(label_dir)
        for f in files:
            lines = open(os.path.join(label_dir, f)).readlines()
            data.append((label, lines))
    return data


def read_20ng():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data = newsgroups_train.data
    labels = newsgroups_train.target


if __name__ == '__main__':
    data = read_amazon('train.csv') + read_amazon('test.csv')
    content_file = os.path.join(data_dir, 'amazon', 'amazon_content.txt')
    write = open(content_file, 'w')
    for d in data:
        _, title, content = d
        write.write(title + '\n')
        write.write(content + '\n')
