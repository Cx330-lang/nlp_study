# -*- encoding: utf-8 -*-
'''
@ Auther: 不正经的程序员<467732983@qq.coom>
@ File: model.py
'''
import tensorflow as tf

class TextCnnModel:
    def __init__(self, config, vocab_szie, word_vector):
        self.config = config
        self.vocab_size = vocab_szie
        self.word_vector = word_vector

        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.labels = tf.placeholder(tf.float32, [None], name='labels')
        self.training = tf.placeholder(tf.bool, name='training')

        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.name_scope("embedding"):
            if self.word_vector:
                embedding_w = tf.Variable(tf.cast(self.word_vector, tf.float32, name='word2vec'), name='embedding_w')
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config['embedding_size']],
                                              initializer=tf.contrib.layers.xavier_initializer())

        embedding_words = tf.nn.embedding_lookup(embedding_w, self.inputs)
        embedding_words_expand = tf.expand_dims(embedding_words, -1)

        pool_outputs = []

        for i, filter_size in enumerate(self.config['filter_sizes']):
            with tf.name_scope("conv_pool-%s"%filter_size):
                filter_shape = [filter_size, self.config['embedding_size'], 1, self.config['num_filters']]

                conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='conv_w')
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config['num_filters']]), name="conv_b")

                conv = tf.nn.conv2d(
                    embedding_words_expand,
                    conv_w,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config['sequence_length'] - filter_size + 1, 1,1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pooled'
                )

                pool_outputs.append(pooled)

        num_total_filters = self.config['num_filters'] * len(self.config['filter_sizes'])

        h_pool = tf.concat(pool_outputs, 3)

        h_pool_flat = tf.reshape(h_pool, [-1, num_total_filters], name="features")

        with tf.name_scope("dropout"):
            h_drop = tf.layers.dropout(h_pool_flat, self.config['drop_rate'], training=self.training)

        with tf.name_scope("output"):
            output_w = tf.get_variable('output_w', shape=[num_total_filters, self.config['num_classes']],
                                       initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config['num_classes']]), name='output_b')

            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name='logits')

            self.scores = tf.nn.softmax(self.logits, name='scores')

            self.predictions = tf.argmax(self.scores, -1, name='predictions')

        self.labels = tf.cast(self.labels, dtype=tf.int32)

        if self.config['focal_loss']:
            losses = self.focal_loss(self.scores, self.labels)
        elif self.config['lmcl_loss']:
            losses = self.lcml_loss(self.logits, self.labels)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)

        self.loss = tf.reduce_mean(losses, name='loss')

        opitimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        trainable_params = tf.trainable_variables()
        gradient = tf.gradients(self.loss, trainable_params)

        gradient_clip, _ = tf.clip_by_global_norm(gradient, self.config["max_grad_norm"])

        self.train_op = opitimizer.apply_gradients(zip(gradient_clip, trainable_params))

    def init_saver(self):
        self.saver = tf.train.Saver()


    def focal_loss(self, preds, labels, gamma=1, alpha=None, epsilon=1e-8):
        labels2d = tf.one_hot(labels, depth=self.config['num_classes'])
        preds = tf.clip_by_value(preds, epsilon, 1-epsilon)

        cross_entropy = - labels2d * tf.log(preds)

        focal_nodulation = (1-preds) ** gamma

        floss = focal_nodulation * cross_entropy

        floss = tf.reduce_sum(floss, -1)

        if alpha:
            alpha = tf.gather(alpha, labels, axis=0)
            floss *= alpha

        return floss

    # 还没弄明白，抽时间弄明白
    def lcml_loss(self, logits, labels, scale=10, margin=0.35):
        labeles_2d = tf.one_hot(labels, depth=self.config['num_classes'], dtype=tf.float32)
        norm_logits = tf.nn.l2_normalize(logits, axis=1)

        new_logits = labeles_2d *(norm_logits - margin) + (1 - labeles_2d) * norm_logits

        new_logits *= scale

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=new_logits, labels=labels)

        return losses





