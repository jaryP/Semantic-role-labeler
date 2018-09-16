import tensorflow as tf

####################################################################################
#LSTM
####################################################################################


def lstm(x, cell, seq_len, backward=False):
    '''
    unroll the lstm dybamically, backward or forward.
    :param x: input data
    :param cell: the cell to unroll
    :param seq_len: the squance length of the data in x
    :param backward: if the data shoul be passed backward
    :return: the results of the pass
    '''

    if backward:
        x = tf.reverse_sequence(input=x, seq_lengths=seq_len, seq_dim=1, batch_dim=0)

    out, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32, swap_memory=True)

    if backward:
        out = tf.reverse_sequence(input=out, seq_lengths=seq_len, seq_dim=1, batch_dim=0)

    return out


def bidirectioal_lstm(x, size, seq_len, sum=False, cell_type='lstm'):
    '''
    create the bidirectional layer
    :param x: input data
    :param size: the cells of the layer
    :param seq_len: the squance length of the data in x
    :param sum: if the output of forward and backward should be summed together
    :param cell_type: the cell type: lstm or gru
    :return: the output of the layer
    '''

    if cell_type == 'lstm':
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, state_is_tuple=True, name='fw')
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=size, state_is_tuple=True, name='bw')
    elif cell_type == 'gru':
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units=size, name='fw')
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units=size, name='bw')
    else:
        assert(False)

    with tf.variable_scope('fw'):
        fw = lstm(x, cell_fw, seq_len)
    with tf.variable_scope('bw'):
        bw = lstm(x, cell_bw, seq_len, backward=True)

    v = cell_fw.variables
    tf.summary.histogram("kernel_fw", v[0])
    tf.summary.histogram("bias_fw", v[1])

    v = cell_bw.variables
    tf.summary.histogram("kernel_bw", v[0])
    tf.summary.histogram("bias_bw", v[1])

    if sum:
        out = fw + bw
    else:
        out = tf.concat((fw, bw), 2)

    return out


def encoder(x, sizes, seq_len, final_out_dim, keep_prob=1.0, sum=False, cells_type='lstm', attention_layers=None,
            heads=None, use_intermediate_states=False, predicate_index=None, layers_type='rnn', ensemble=False):
    '''
    Build up the decoder
    '''

    mask = tf.sequence_mask(seq_len)
    mask = tf.expand_dims(tf.to_float(mask), -1)

    def get_predicate_emb(input):
        assert mask is not None
        shape = tf.shape(input)
        idx = tf.stack([tf.range(shape[0], dtype=tf.int32),
                        predicate_index], axis=1)

        predicate_emb = tf.gather_nd(input, idx)
        predicate_emb = tf.tile(tf.expand_dims(predicate_emb, 1),
                                (1, shape[1], 1))
        out = tf.concat([input, predicate_emb], axis=2)
        out *= mask
        return out

    def layer_forward(x, layer_n):
        if predicate_index is not None:
            input = get_predicate_emb(x)
        else:
            input = x

        with tf.device('/cpu:0'):
            with tf.variable_scope('variables', reuse=tf.AUTO_REUSE):
                w_out = tf.get_variable(name='out_weights_{}'.format(layer_n), shape=[input.shape[-1], final_out_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram('out_weights_{}'.format(layer_n), w_out)

            out = batch_matmul(input, w_out)
        return out

    if attention_layers is None:
        attention_layers = [0] * len(sizes)
    else:
        assert (mask is not None)

        if isinstance(attention_layers, int):
            zeros = [0] * len(sizes)
            zeros[attention_layers] = 1
            attention_layers = zeros

        elif attention_layers == 'all':
            attention_layers = [1] * len(sizes)

        assert(len(attention_layers) == len(sizes))

        if heads is None:
            heads = 16

    if not isinstance(sizes, (list, tuple)):
            sizes = [sizes]

    out = 0

    next_input = x
    layer_out = x
    l = 0

    with tf.name_scope('lstm_encoder'):
        for l, c in enumerate(sizes):
            with tf.variable_scope('layer_{}'.format(l)):
                with tf.variable_scope('layer_{}'.format(l)):

                    if layers_type == 'rnn':
                        layer_out = bidirectioal_lstm(next_input, c, seq_len, sum, cell_type=cells_type)
                        c = c * 2
                    else:
                        layer_out = ffn_layer(next_input, c)# * mask

                    layer_out = tf.nn.dropout(layer_out, keep_prob)

                if attention_layers[l]:
                    with tf.device('/cpu:0'):
                        layer_out = multihead_attention_layer(layer_out, c, heads, keep_prob, l)

                if use_intermediate_states or ensemble:
                    logits = layer_forward(layer_out, l)
                    tf.summary.histogram('out_weights_{}'.format(l), logits)
                    out += logits

                if not ensemble:
                    next_input = layer_out

    if out == 0:
        out = layer_forward(layer_out, l)

    tf.summary.histogram("forward_output", out)

    return out

####################################################################################
#FFD
####################################################################################


def ffn_layer(inputs, size):
    '''
    The FFW layer used instead of RNN
    '''
    d = inputs.shape[-1]
    w = tf.get_variable(shape=[d, size], name='w1')
    b = tf.get_variable(shape=[size], name='b1')
    tf.summary.histogram('w1', w)
    tf.summary.histogram('b1', b)

    w1 = tf.get_variable(shape=[size, d], name='w2')
    b1 = tf.get_variable(shape=[d], name='b2')
    tf.summary.histogram('w2', w1)
    tf.summary.histogram('b2', b1)

    h = tf.nn.relu(batch_matmul(inputs, w))

    output = batch_matmul(h, w1)

    return output

####################################################################################
#ATTENTIION MECHANISM
####################################################################################


def attention_head(q, k, v, d, keep_prob=1.0):
    '''
    A single attention head calculation
    :param q: query
    :param k: key
    :param v: value
    :param d: size of the head
    :param keep_prob: drop out probability
    '''
    num = tf.matmul(q, k, transpose_b=True) * (d ** -0.5)
    w = tf.nn.softmax(num, name="attention_weights")
    out = tf.matmul(w, v)
    tf.summary.histogram('attention_out', out)
    out = tf.nn.dropout(out, keep_prob)
    return out


def multihead_attention_layer(x, d, num_heads, keep_prob=1.0, layer=0):
    '''
    Multi head attention mechanism
    '''
    def split_head(input):
        old_shape = tf.shape(input)
        new_shape = tf.concat([old_shape[:-1], [num_heads, d//num_heads]], -1)
        input = tf.reshape(input, new_shape)
        return tf.transpose(input, [0, 2, 1, 3])

    def combine_heads(input):
        input = tf.transpose(input, [0, 2, 1, 3])
        old_shape = tf.shape(input)
        new_shape = tf.concat([old_shape[:-2], [old_shape[-2] * old_shape[-1]]], -1)
        return tf.reshape(input, new_shape)

    assert(d % num_heads == 0)
    shape = x.get_shape()
    with tf.name_scope('multihead_attention_layer_{}'.format(layer)):
        with tf.name_scope('input_transform'):
            with tf.variable_scope('variables', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('attention_layer_{}'.format(layer)):
                    w_projector = tf.get_variable(shape=[shape[-1], d*3], name='projector_weights',
                                                  initializer=tf.contrib.layers.xavier_initializer())
                    b_projector = tf.get_variable(shape=[d*3], name='projector_bias', initializer=tf.zeros_initializer)

                    tf.summary.histogram('projector_w', w_projector)
                    tf.summary.histogram('projector_b', b_projector)

                    wo_projector = tf.get_variable(shape=[d, d], name='projector_weights_o',
                                                   initializer=tf.contrib.layers.xavier_initializer())
                    bo_projector = tf.get_variable(shape=[d], name='projector_bias_o',
                                                   initializer=tf.zeros_initializer)

                    tf.summary.histogram('projector_w_o', wo_projector)
                    tf.summary.histogram('projector_b_o', bo_projector)

            proj = batch_matmul(x, w_projector, b_projector)

        split_dim = d//num_heads

        with tf.name_scope('output_transform'):
            with tf.name_scope('heads'):
                q, k, v = tf.split(proj, [d, d, d], axis=-1)

                q = split_head(q)
                k = split_head(k)
                v = split_head(v)

                c = attention_head(q, k, v, split_dim, keep_prob=keep_prob)
                c = combine_heads(c)

            out = batch_matmul(c, wo_projector, bo_projector)

        tf.summary.histogram('multi_out', out)
        out = tf.nn.dropout(out, keep_prob)

    return out

####################################################################################
#UTILS
####################################################################################


def batch_matmul(x, w, b=None):
    '''
    General matmul that works for n dimensions
    '''
    with tf.name_scope('batch_matmul'):
        in_shapes = tf.shape(x)
        out_shape = tf.shape(w)
        x = tf.reshape(x, [-1, in_shapes[-1]])
        r = tf.matmul(x, w)
        if b is not None:
            r += b
        r = tf.reshape(r, tf.concat([in_shapes[:-1], out_shape[1:]], 0))
    return r



