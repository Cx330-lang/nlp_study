# -*- encoding: utf-8 -*-

import os
import random
from itertools import chain
from collections import Counter

import jieba
import gensim
import numpy as np
import tensorflow as tf


class TrainData(object):
    def __init__(self, config):
        self.__output_path = config['output_path']
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self._sequence_length = config['sequence_length']
        self._batch_size = config['batch_size']
        self._vocab_size = config['vocab_size']

        self._use_word = config['use_word']
        self._word_vectors_path = config['word_vectors_path']
        self._embedding_size = config['embedding_size']
        self.word_vectors = None

    @staticmethod
    def read_data(file_path):
        inputs = []
        labels = []

        with tf.gfile.GFile(file_path, 'r') as fr:
            for line in fr.readlines():
                item = line.strip().split('\t')

                if item[1].strip() == '':
                    continue

                inputs.append(item[1])
                labels.append(item[0])

        return inputs, labels


    def get_word_vector(self, vocab):
        """
        加载词向量，并获得相应的词向量的矩阵
        :param vocab: 训练集所含有的词
        :return: 词向量矩阵
        """

        word_vectors = (1 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self._embedding_size) - 1))

        if os.path.splitext(self._word_vectors_path)[-1] == '.bin':
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self._word_vectors_path)

        for i in range(len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(vocab[i] + "不存在词向量中")

        return word_vectors

    def trans_to_index(self, inputs, word_to_index):
        """
        将输入转换为索引表示
        :param inputs:
        :param word_to_index:
        :return:
        """

        unk_id = word_to_index.get("[UNK]")
        if self._use_word:
            input_ids = [[word_to_index.get(word, unk_id) for word in jieba.lcut(inp)] for
                         inp in inputs]
        else:
            input_ids = [[word_to_index.get(word, unk_id) for word in inp] for inp in inputs]

        return input_ids

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        labels_idx = [label_to_index[label] for label in labels]

        return labels_idx

    def padding(self, input_ids, word_to_index):
        pad_id = word_to_index.get("[PAD]")

        pad_input_ids = []

        for input_id in input_ids:
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [pad_id] * (self._sequence_length - len(input_id)))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])

        return pad_input_ids

    def gen_data(self, file_path, is_trianing=True):
        """
        生成数据
        :param file_path:
        :param is_trianing:
        :return:
        """

        inputs, labels = self.read_data(file_path)

        if is_trianing:
            if self._use_word:
                words = [[word for word in jieba.lcut(inp) if word.strip() != ""] for inp in inputs]
            else:
                words = [[word for word in inp if word.strip() != ""] for inp in inputs]

            words = list(chain(*words))
            word_count = Counter(words)

            word_count_sort = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:self._vocab_size]

            unique_word = ["[PAD]"] + ["[UNK]"] + [item[0] for item in word_count_sort]

            if self._word_vectors_path != "":
                self.word_vectors = self.get_word_vector(unique_word)

            word_to_index = dict(zip(unique_word, range(len(unique_word))))

            with tf.gfile.GFile(os.path.join(self.__output_path, "word_to_index.txt"), 'w') as fw:
                word_save = [key + "\t" + str(value) for key,value in word_to_index.items()]
                fw.write("\n".join(word_save))

            unique_label = list(set(labels))
            uni_label = sorted(unique_label)
            label_to_index = dict(zip(uni_label, range(len(unique_label))))

            with tf.gfile.GFile(os.path.join(self.__output_path, "label_to_index.txt"), 'w') as fw:
                label_save = [key + "\t" + str(value) for key, value in label_to_index.items()]
                fw.write('\n'.join(label_save))

        else:
            word_to_index = {}
            with tf.gfile.GFile(os.path.join(self.__output_path, "word_to_index.txt"), 'r') as fr:
                for line in fr:
                    item = line.strip().split('\t')
                    word_to_index[item[0]] = int(item[1])

            label_to_index = {}
            with tf.gfile.GFile(os.path.join(self.__output_path, "label_to_index.txt"), 'r') as fr:
                for line in fr:
                    item = line.strip().split('\t')
                    label_to_index[item[0]] = int(item[1])

        input_ids = self.trans_to_index(inputs, word_to_index)

        input_ids = self.padding(input_ids, word_to_index)

        label_ids = self.trans_label_to_index(labels, label_to_index)

        return input_ids, label_ids, label_to_index, len(word_to_index)

    def next_batch(self, input_ids, label_ids, is_training=True):
        """
        生成batch 数据
        :param input_ids:
        :param label_ids:
        :param is_training:
        :return:
        """

        z = list(zip(input_ids, label_ids))
        random.shuffle(z)
        input_ids, label_ids = zip(*z)

        num_batches = len(input_ids) // self._batch_size

        if not is_training:
            num_batches += 1

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size

            batch_input_ids = input_ids[start:end]
            batch_label_ids = label_ids[start:end]

            yield dict(input_ids=batch_input_ids,
                        label_ids=batch_label_ids)











