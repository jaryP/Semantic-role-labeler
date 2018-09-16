import os
import datetime
import tensorflow as tf
import numpy as np
import json
import codice.nn_utils as nn
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix as cm


class MLP:
    def __init__(self, dataset, **kwargs):

        #default parameter are the best model
        self.hyper = {
                      "rnn_cell_type": "lstm",
                      "layers_type": "rnn",
                      "layers_size": [512, 512, 512],
                      "use_intermediate_states": 0,
                      "w_emb": 100,
                      "lr": 1e-3,
                      "dropout": 0.2,
                      "pos_emb": 16,
                      "incremental": 1,
                      "incorporate_predicate_emb": 0,
                      "use_attention": 1,
                      "layers_attention": -1,
                      "attention_heads": 16,
                      "batch_size": 40,
                      "to_load": 1,
                      "gradient_clip": 1,
                      "log_dir":
                      "/media/jary/DATA/Uni/NLP/17-18/HWs/HW3/logs/report/final",
                      "epochs": 20,
                      'log_dir': None, 'epochs': 5, 'n_words': dataset.words_len,
                      'n_pos': dataset.pos_len, 'n_roles': dataset.roles_len, 'n_lemmas': dataset.lemma_len}

        self.hyper.update(**kwargs)

        self.dataset = dataset
        self.null = dataset.role_id_map['NULL']

        log_dir = self.hyper['log_dir']
        if log_dir is None:
            log_dir = os.path.join('../logs', datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
        self.log_dir = log_dir

        self.is_training = False
        self.transition_parameters = None

        self.loss, self.opt, self.prediction = self._build_net()

        with tf.variable_scope('init'):
            self._session = tf.Session()
            init = tf.global_variables_initializer()
            self._session.run(init)

        self.back_dict = {'epoch': 1}
        self.back_dict.update(self.hyper)

        if self.hyper['to_load']:
            self.load()

        self.back_dict.update({'epochs': kwargs.get('epochs')})

        print({k: v for k, v in self.back_dict.items() if k != 'results'})

    def load(self, epoch=None, best_one=False):
        '''
        load a saved network
        '''
        try:
            saver = tf.train.Saver()

            with open(os.path.join(self.log_dir, 'back_dict.json'), 'r') as f:
                data = f.read()
                js = json.loads(data)
                self.back_dict.update(js)

            if epoch is None:
                epoch = self.back_dict['epoch']-1

            if best_one:
                best_score = 0
                epoch = 0
                for k, v in self.back_dict['results'].items():
                    score = v['score'][0]['labeled']['f1']
                    if best_score < score:
                        epoch = k
                        best_score = score
                print('Loading best model at epoch {}'.format(epoch))
            saver.restore(self._session, os.path.join(self.log_dir, str(epoch), 'model.ckpt'))
            print('Model loaded')

        except Exception as e:
            print('Model not loaded')
            print(e)

    def train(self):
        '''
        Train the network on train dataset and evaluate it after each epoch on dev dataset
        '''

        if self.back_dict['tolerance'] == 0 or self.back_dict['epoch'] > self.back_dict['epochs']:
            return

        batch_size = self.hyper['batch_size']
        print_step = int((len(self.dataset.train_indexes) / (10 * batch_size)))

        writer = tf.summary.FileWriter(self.log_dir, self._session.graph)
        saver = tf.train.Saver()

        merged = tf.summary.merge_all()
        loss_summary = tf.summary.merge([self.loss_summary])

        best_score = 0

        if 'results' in self.back_dict:
            for k, v in self.back_dict['results'].items():
                score = v['score'][0]['labeled']['f1']
                if best_score < score:
                    best_score = score

        for e in range(self.back_dict['epoch'], self.hyper['epochs'] + 1):

            self.dataset.training = True

            print('#' * 50)
            print('Epoch: {}, best score so far: {} with tolerance: {}'.format(e, best_score,
                                                                               self.back_dict['tolerance']))

            batches = 0
            losses = []

            batcher = self.dataset.get_train_batch if self.hyper['incremental'] == 0 else \
                self.dataset.get_train_batch_incremental

            for j, batch in enumerate(batcher(batch=batch_size)):

                feed = self.batch_to_feed(batch)

                loss, _, m, step, ls = self._session.run([self.loss, self.opt, merged, self.global_step, loss_summary],
                                                         feed_dict=feed)
                batches += 1
                losses.append(float(loss))

                if (j+1) % 10 == 0:
                    writer.add_summary(ls, global_step=step)

                if (j + 1) % print_step == 0:
                    writer.add_summary(m, global_step=step)
                    print('Step:', (j + 1) / print_step)
                    print('\tLoss mean on last {} batches: {}'.format(batches, np.mean(losses[-batches:])))

            print('#' * 50)

            self.dataset.training = False

            saver.save(self._session, os.path.join(self.log_dir, str(e), 'model.ckpt'))
            self.back_dict['epoch'] = e + 1

            res = self.get_dataset_score(dataset='dev')

            if res['labeled']['f1'] > best_score:
                best_score = res['labeled']['f1']
                self.back_dict['tolerance'] = 5
            else:
                self.back_dict['tolerance'] -= 1

            print('Scores after epoch #{}'.format(e), res)
            results = self.back_dict.get('results', {})
            r, rl = results.get('scores', []), results.get('losses', [])
            r.append(res)
            rl.append(losses)
            results[e] = {'score': r, 'losses': rl}
            self.back_dict['results'] = results

            with open(os.path.join(self.log_dir, 'back_dict.json'), 'w') as f:
                f.write(json.dumps(self.back_dict))

            if self.back_dict['tolerance'] == 0:
                break

        writer.close()

    def batch_to_feed(self, batch, training=True):
        '''
        Given a batch create the feed dict to be passed to the network
        '''
        (word_index, pos_index, predicate_index, labels, lengths, w2v_index, lemma_index) = batch
        feed_dict = {
            self.w_in: word_index,
            self.p_in: pos_index,
            self.predicate_index: predicate_index,
            self.labels: labels,
            self.lengths: lengths,
            self.w2v: w2v_index,
            self.lemma_eb: lemma_index,
            self.is_training: 1 if training else 0
        }

        return feed_dict

    def forward(self, batch):
        '''
        calculate the forwad pass of the network
        '''
        return self._session.run(self.prediction, feed_dict=self.batch_to_feed(batch, False))

    def get_dataset_score(self, dataset='dev'):
        '''
        Get the score associated for a given dataset: train or dev
        '''
        correct_labeled = 0
        correct_unlabeled = 0
        num_predicted = 0
        num_gold = 0

        for batch in self.dataset.get_all_dataset(dataset, batch=self.hyper['batch_size']):

            val = self.forward(batch)
            y = batch[3]
            s = y.shape

            for b in np.arange(s[0]):
                for ts in np.arange(batch[4][b]):
                    p = val[b][ts]
                    g = y[b][ts]
                    if p != self.null:
                        num_predicted += 1
                        if p == g:
                            correct_labeled += 1
                            correct_unlabeled += 1
                        elif g != self.null:
                            correct_unlabeled += 1

                    if g != self.null:
                        num_gold += 1

        precision = correct_labeled / num_predicted
        recall = correct_labeled / num_gold
        f1 = (2 * precision * recall /
                      (precision + recall))

        un_precision = correct_unlabeled / num_predicted
        un_recall = correct_unlabeled / num_gold
        un_f1 = (2 * un_precision * un_recall /
                        (un_precision + un_recall))

        return {'labeled': {'precision': precision*100, 'recall': recall*100, 'f1': f1*100},
                'unlabeled': {'precision': un_precision*100, 'recall': un_recall*100, 'f1': un_f1*100}}

    def study_results(self):
        '''
        function that create and plot or print different evaluation on the dev dataset
        '''

        def calculate_score(res):
            p = 0 if 'predicted' not in res or res['predicted'] == 0 else res['correct_labeled'] / res['predicted']
            r = 0 if 'gold' not in res or res['gold'] == 0 else res['correct_labeled'] / res['gold']
            f1 = 0 if p == r == 0 else (2 * p * r / (p + r))

            return p * 100, r * 100, f1 * 100

        def save_confusion_matrix(g, p, name='confusion_matrix'):
            classes = [k for k, v in self.dataset.role_id_map.items() if v in set(g) | set(p)]
            plt.figure(figsize=(18, 18))
            confusion_matrix = cm(g, p)
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            confusion_matrix[np.isnan(confusion_matrix)] = 0
            confusion_matrix *= 100
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("confusion matrix")
            thresh = confusion_matrix.max() / 2.

            for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
                if confusion_matrix[i, j] > 0:
                    plt.text(j, i, format(confusion_matrix[i, j], '.1f'),
                             horizontalalignment="center",
                             color="white" if confusion_matrix[i, j] > thresh else "black", fontsize=12)

            plt.tight_layout()
            plt.xlabel('Predicted label')
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.yticks(fontsize=12)
            plt.savefig(os.path.join(self.log_dir, '{}.png'.format(name)), bbox_inches='tight')

        def save_distance_plot(d):

            keys_map = {(0, 0): 0, (1, 5): 1, (6, 10): 3, (11, 15): 4, (16, 30): 5, (31, float('inf')): 6}
            keys_map_occurrence = defaultdict(int)

            distances = {v: defaultdict(int) for v in keys_map.values()}

            for k, v in d.items():
                if len(v) == 0:
                    continue
                key = None
                for km, vm in keys_map.items():
                    if km[0] <= k <= km[1]:
                        key = vm
                        break

                keys_map_occurrence[key] += distances_from_predicate[k]
                distances[key]['predicted'] += v['predicted']
                distances[key]['correct_labeled'] += v['correct_labeled']
                distances[key]['gold'] += v['gold']

            f1s = []
            pcs = []
            rcs = []
            total = sum(v for v in keys_map_occurrence.values())

            for k, v in distances.items():
                p, r, f1 = calculate_score(v)
                f1s.append(f1)
                pcs.append(p)
                rcs.append(r)

            x = range(len(f1s))
            plt.title("Score varying distance from predicate")
            plt.plot(x, f1s, label='f1')
            plt.plot(x, pcs, label='precision')
            plt.plot(x, rcs, label='recall')
            plt.legend()
            plt.tight_layout()
            plt.xlabel('Distance from predicate')
            plt.ylabel('F1 score')
            # xlabel = ['{}\n{:.2f}%'.format(c, (keys_map_occurrence[v] / total) * 100) for c, v in keys_map.items()]
            xlabel = ['{}'.format(c) for c, v in keys_map.items()]
            plt.xticks(x, xlabel, rotation=45)
            plt.savefig(os.path.join(self.log_dir, 'score_varying_distance.png'), bbox_inches='tight')
            plt.close()

        def save_len_plot(d):

            keys_map = {(1, 5): 0, (6, 10): 1, (11, 15): 2, (16, 30): 3, (31, 50): 4, (51, float('inf')): 5}
            keys_map_occurrence = defaultdict(int)

            distances = {v: defaultdict(int) for v in keys_map.values()}

            for k, v in d.items():
                if len(v) == 0:
                    continue
                key = None
                for km, vm in keys_map.items():
                    if km[0] <= k <= km[1]:
                        key = vm
                        break

                keys_map_occurrence[key] += sentences_length[k]
                distances[key]['predicted'] += v['predicted']
                distances[key]['correct_labeled'] += v['correct_labeled']
                distances[key]['gold'] += v['gold']

            f1s = []
            pcs = []
            rcs = []
            total = sum(v for v in keys_map_occurrence.values())

            for k, v in distances.items():
                p, r, f1 = calculate_score(v)
                f1s.append(f1)
                pcs.append(p)
                rcs.append(r)

            x = range(len(f1s))
            plt.title("Score varying sentence length")
            plt.plot(x, f1s, label='f1')
            plt.plot(x, pcs, label='precision')
            plt.plot(x, rcs, label='recall')
            plt.legend()
            plt.tight_layout()
            plt.xlabel('Score based on sentence length')
            plt.ylabel('F1 score')
            # xlabel = ['{}\n{:.2f}%'.format(c, (keys_map_occurrence[v] / total) * 100) for c, v in keys_map.items()]
            xlabel = ['{}'.format(c) for c, v in keys_map.items()]
            plt.xticks(x, xlabel, rotation=45)
            plt.savefig(os.path.join(self.log_dir, 'score_varying_sent_len.png'), bbox_inches='tight')
            plt.close()

        self.load(best_one=True)

        score_per_lengts = dict()
        distance_from_predicate_scores = dict()
        pos_scores = dict()

        confusion_matrix = np.zeros((self.hyper['n_roles'], self.hyper['n_roles']))

        pred = []
        gold = []
        pos_dictionary = {v: k for k, v in self.dataset.pos_id_map.items()}
        role_dictionary = {v: k for k, v in self.dataset.role_id_map.items()}
        distances_from_predicate = []
        sentences_length = []

        for batch in self.dataset.get_all_dataset('dev', batch=self.hyper['batch_size']):

            (_, pos_index, predicate_index, labels, lengths, _, _) = batch
            feed = self.batch_to_feed(batch, training=False)
            val = self._session.run(self.prediction, feed_dict=feed)
            y = labels
            s = y.shape

            for b in np.arange(s[0]):
                sent_len = lengths[b]
                sentences_length.append(sent_len)
                pred_len = score_per_lengts.get(sent_len, defaultdict(int))
                b_pred_index = predicate_index[b]

                for ts in np.arange(sent_len):
                    p = val[b][ts]
                    g = y[b][ts]

                    dis = abs(ts - b_pred_index)
                    distances_from_predicate.append(dis)
                    pred_dis = distance_from_predicate_scores.get(dis, defaultdict(int))

                    super_role = 'NULL'
                    if 'A0' in role_dictionary[g]:
                        super_role = 'A0'
                    elif 'A1' in role_dictionary[g]:
                        super_role = 'A1'
                    elif 'A2' in role_dictionary[g]:
                        super_role = 'A2'
                    elif 'AM' in role_dictionary[g]:
                        super_role = 'AM'

                    pred_pos = pos_scores.get(pos_dictionary[pos_index[b][ts]] + '-' + super_role, defaultdict(int))

                    confusion_matrix[g, p] += 1
                    confusion_matrix[p, g] += 1

                    pred.append(p)
                    gold.append(g)

                    if p != self.null:
                        pred_len['predicted'] += 1
                        pred_dis['predicted'] += 1
                        pred_pos['predicted'] += 1

                        if p == g:
                            pred_len['correct_labeled'] += 1
                            pred_dis['correct_labeled'] += 1
                            pred_pos['correct_labeled'] += 1

                    if g != self.null:
                        pred_len['gold'] += 1
                        pred_dis['gold'] += 1
                        pred_pos['gold'] += 1

                    distance_from_predicate_scores[dis] = pred_dis
                    pos_scores[pos_dictionary[pos_index[b][ts]] + '-' + super_role] = pred_pos

                score_per_lengts[sent_len] = pred_len

        distances_from_predicate = Counter(distances_from_predicate)
        sentences_length = Counter(sentences_length)

        save_len_plot(score_per_lengts)
        save_distance_plot(distance_from_predicate_scores)

        print('pos scores')
        subset = {}
        for k, v in pos_scores.items():
            key = k[:2]
            if key not in subset:
                subset[key] = {'A0': defaultdict(int), 'A1': defaultdict(int), 'A2': defaultdict(int),
                               'AM': defaultdict(int), 'NULL': defaultdict(int)}

            subk = k[-2:] if 'NULL' not in k else 'NULL'

            subset[key][subk]['predicted'] += v['predicted']
            subset[key][subk]['correct_labeled'] += v['correct_labeled']
            subset[key][subk]['gold'] += v['gold']

        for k, v in subset.items():
            for k1, v1 in v.items():
                p, r, f1 = calculate_score(v1)
                print('{} & {:.2f} & {:.2f} & {:.2f}'.format(k1, p, r, f1) + ' \\\\ \cline{2-5}')
        save_confusion_matrix(gold, pred)

    def _build_net(self):
        '''
        The function that build the network
        :return: loss, train operation and forward pass
        '''

        batch_size = None
        num_words = None

        with tf.variable_scope('variables', reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('inputs'):
            self.w_in = tf.placeholder(tf.int32, shape=[batch_size, num_words], name='word_emb_input')
            self.p_in = tf.placeholder(tf.int32, shape=[batch_size, num_words], name='pos_emb_input')
            self.predicate_index = tf.placeholder(tf.int32, shape=[batch_size, ], name='predicates_indexes')
            self.labels = tf.placeholder(tf.int32, shape=[batch_size, num_words], name='labels_input')
            self.lengths = tf.placeholder(tf.int32, shape=[batch_size], name='sencences_lengths')
            self.w2v = tf.placeholder(tf.int32, shape=[batch_size, num_words], name='w2v_embedding')
            self.lemma_eb = tf.placeholder(tf.int32, shape=[batch_size, num_words], name='lemma_emb_input')
            self.is_training = tf.placeholder(tf.float32, shape=[], name='use_dropout_flag')

            keep_prob = 1 - (self.hyper['dropout'] * self.is_training)

            words = tf.shape(self.labels)[1]

            predicate_flag = tf.expand_dims(tf.one_hot(self.predicate_index, words, dtype=tf.float32),
                                            axis=-1)

        with tf.name_scope('Embeddings'):
            with tf.device('/cpu:0'):

                pos_embeddings = tf.Variable(
                    tf.random_normal([self.hyper['n_pos'], self.hyper['pos_emb']], dtype=tf.float32, stddev=0.1
                                     ), name='pos_emb_matrix')

                ##WORDS EMB
                words_embeddings = tf.Variable(
                    tf.random_normal([self.hyper['n_words'], self.hyper['w_emb']], dtype=tf.float32, stddev=0.1),
                    name='words_emb_matrix', trainable=True)

                w2v_embeddings = tf.Variable(initial_value=tf.constant(self.dataset.get_w2v_embeddings(),
                                                                       dtype=tf.float32), trainable=False,
                                             name='w2v_emb')

                #LEMMAS EMB
                nasari_emb = tf.Variable(initial_value=tf.constant(self.dataset.get_nasari_embeddings(),
                                                                   dtype=tf.float32), trainable=True, name='nasari_emb')


                padding_nasari = tf.Variable(initial_value=tf.zeros(shape=[1, 300]), trainable=False, dtype=tf.float32)

        with tf.name_scope('input_construction'):
            p_emb = tf.nn.embedding_lookup(pos_embeddings, self.p_in)

            padded_nasari_emb = tf.concat([padding_nasari, nasari_emb], axis=0)
            lemma_emb = tf.nn.embedding_lookup(padded_nasari_emb, self.lemma_eb)

            w_emb = tf.concat([tf.nn.embedding_lookup(words_embeddings, self.w_in),
                                   tf.nn.embedding_lookup(w2v_embeddings, self.w2v)], -1)

            if self.hyper['layers_type'] == 'ffw':
                inp_shape = tf.shape(w_emb)
                position_embedding = tf.get_variable(shape=[500, w_emb.shape[-1]],
                                                     dtype=tf.float32,
                                                     name='position_embedding')
                indices = tf.range(inp_shape[1])[None, :]
                pos_emb = tf.gather(position_embedding, indices)
                pos_emb = tf.tile(pos_emb, [inp_shape[0], 1, 1])
                w_emb += pos_emb

            input = tf.concat([w_emb, p_emb, lemma_emb, predicate_flag], 2)

        attention_layers = self.hyper['layers_attention'] if self.hyper['use_attention'] else None

        predicate_index = None
        if self.hyper['incorporate_predicate_emb']:
            predicate_index = self.predicate_index

        logits = nn.encoder(input, sizes=self.hyper['layers_size'], seq_len=self.lengths, keep_prob=keep_prob,
                            cells_type=self.hyper['rnn_cell_type'], attention_layers=attention_layers,
                            use_intermediate_states=self.hyper['use_intermediate_states'], ensemble=self.hyper['ensemble'],
                            final_out_dim=self.hyper['n_roles'], heads=self.hyper['attention_heads'],
                            predicate_index=predicate_index, layers_type=self.hyper['layers_type'])

        if self.hyper['crf']:
            with tf.name_scope('crf'):

                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(logits, self.labels, self.lengths)
                self.trans_params = trans_params
                loss = -log_likelihood
                forward, _ = tf.contrib.crf.crf_decode(logits, self.trans_params, self.lengths)

        else:
            with tf.name_scope('cross_entropy'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
                forward = tf.argmax(tf.nn.softmax(logits, 2), 2)
        loss = tf.reduce_mean(loss)

        self.loss_summary = tf.summary.scalar('loss', loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hyper['lr'])

            grads, vars = zip(*optimizer.compute_gradients(loss))
            #
            if self.hyper['gradient_clip']:
                grads, _ = tf.clip_by_global_norm(grads, 20.0)
            grads_and_vars = list(zip(grads, vars))

            for grad, var in grads_and_vars:
                if grad is not None:

                    tf.summary.histogram(var.name + '/gradient_norm', tf.norm(grad))
                    tf.summary.scalar(var.name+'/gradient_mean', tf.reduce_mean(tf.abs(grad)))
                    tf.summary.scalar(var.name + '/gradient_norm', tf.norm(grad))

            train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        return loss, train_op, forward
