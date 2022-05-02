# -*- encoding: utf-8 -*-

import json
import os
import argparse
import time
import sys
import datetime
import tensorflow as tf


from model import TextCnnModel
from dataProcess import TrainData
from metrics import mean, get_multi_metrics

tf.set_random_seed(1234)

def model_tiem():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M"))


class Trainer(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as fr:
            self.config = json.load(fr)

        self.data_obj = self.load_data()
        self.t_in_ids, self.t_lab_ids, lab_to_idx, vocab_size = self.data_obj.gen_data(self.config['train_data'])

        self.e_in_ids, self.e_lab_ids, lab_to_idx, vocab_size = self.data_obj.gen_data(self.config['valid_data'])

        self.label_list = [value for key, value in lab_to_idx.items()]

        self.model = TextCnnModel(self.config, vocab_size, self.data_obj.word_vectors)

    def load_data(self):
        data_obj = TrainData(self.config)

        return data_obj

    def train(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0

            start_time = time.time()

            for epoch in range(self.config['epochs']):
                for batch in self.data_obj.next_batch(self.t_in_ids, self.t_lab_ids):
                    feed_dict = {
                        self.model.inputs: batch['input_ids'],
                        self.model.labels: batch['label_ids'],
                        self.model.training: True
                    }
                    _, loss, predictions = sess.run([self.model.train_op, self.model.loss, self.model.predictions], feed_dict=feed_dict)

                    if current_step % self.config['log_every'] == 0:
                        acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch['label_ids'], labels=self.label_list)

                        print("trian: step:{}, loss: {}, acc:{}, recall:{}, pred:{}, f_beta:{}".format(
                            current_step, loss, acc, recall, prec, f_beta
                        ))

                    current_step += 1

                    if self.data_obj and current_step % self.config['eval_every'] == 0:
                        eval_losses = []
                        eval_preds = []
                        eval_labels = []

                        for eval_batch in self.data_obj.next_batch(self.e_in_ids, self.e_lab_ids, False):
                            feed_dict = {
                                self.model.inputs: eval_batch['input_ids'],
                                self.model.labels: eval_batch['label_ids'],
                                self.model.training: False
                            }

                            eval_loss, eval_predictions = sess.run([self.model.loss, self.model.predictions],
                                                                   feed_dict=feed_dict)

                            eval_losses.append(eval_loss)
                            eval_preds.extend(eval_predictions)
                            eval_labels.extend(eval_batch['label_ids'])

                            acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_preds, true_y=eval_labels, labels=self.label_list)

                            print("\n")
                            print("eval:  loss: {}, acc:{}, recall:{}, pred:{}, f_beta:{}".format(
                                 mean(eval_losses), acc, recall, prec, f_beta
                            ))

                        if current_step % self.config["checkpoint_every"] == 0:
                            save_path = self.config['ckpt_model_path']
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            model_save_path = os.path.join(save_path, self.config['model_name'])

                            self.model.saver.save(sess, model_save_path, global_step=current_step)

                end_time = time.time()
                print('total trian time:', end_time - start_time)


if __name__ == '__main__':
    trainer =Trainer("config.json")

    trainer.train()













