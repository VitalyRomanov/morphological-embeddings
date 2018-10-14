import tensorflow as tf
import numpy as np

# val = np.random.randn(3,3)
# print(val)


def kld(m1, m2, log_v1, log_v2, emb_size):
    # print("KLD inp: ", m1.shape, m2.shape, log_v1.shape, log_v2.shape)
    # a = tf.square(m2 - m1) * tf.exp(-log_v2)
    # print(a.shape)
    # return tf.reduce_sum(log_v1 - log_v2, axis=1) - \
    #             emb_size + \
    #             tf.reduce_sum(tf.exp(log_v2 - log_v1), axis=1) + \
    #             tf.reduce_sum(tf.square(m1 - m2) * tf.exp(-log_v1), axis=1)

    return tf.reduce_sum(log_v2 - log_v1, axis=1) + \
                tf.reduce_sum(tf.exp(log_v1 - log_v2), axis=1) + \
                tf.reduce_sum(tf.square(m2-m1)*tf.exp(-log_v2))

def model(emb_size = 300, context_p_negative = 9, voc_size = 10000):

    context_len = int((context_p_negative - 1) / 2)

    # word_ids = tf.placeholder(shape=(None,context_p_negative), dtype=tf.int32)
    word_ids = tf.placeholder(shape=(context_p_negative, 1), dtype=tf.int32)

    emb_mean = tf.Variable(tf.random_normal(shape=(voc_size, emb_size)), dtype=tf.float32)
    emb_var = tf.Variable(tf.random_normal(shape=(voc_size, emb_size)), dtype=tf.float32)

    get_mean = tf.squeeze(tf.nn.embedding_lookup(emb_mean, word_ids))
    get_var = tf.squeeze(tf.nn.embedding_lookup(emb_var, word_ids))

    word_mean = tf.reshape(get_mean[0], (1,-1))
    word_log_var = tf.reshape(get_var[0], (1,-1))

    print(get_mean.shape, word_mean.shape)

    context_mean = get_mean[1:context_len+1]
    context_log_var = get_var[1:context_len+1]
    # context_vect = tf.reduce_sum(tf.nn.relu(tf.concat([context, word_mean], axis=1)), axis=0)
    context_vect = tf.reduce_sum(
                        tf.nn.dropout(
                            tf.layers.dense(
                                            tf.concat([context_mean, tf.tile(word_mean, (context_len,1))], axis=1),
                                            emb_size,
                                            activation=tf.nn.relu
                                            ),
                            .7),
                        axis=0, keepdims=True)

    print(context_vect.shape)
    word_contextual_mean = tf.nn.dropout(tf.layers.dense(context_vect, emb_size, activation=None), .7)
    word_contextual_log_var = tf.nn.dropout(tf.layers.dense(context_vect, emb_size, activation=None), .7)

    negative_mean = get_mean[context_len+1:]
    negative_log_var = get_var[context_len+1:]

    # a = tf.reduce_mean(kld(word_contextual_mean, context_mean, word_contextual_log_var, context_log_var, emb_size), axis=0)
    # print(a.shape)
    # a = tf.reduce_mean(kld(word_contextual_mean, negative_mean, word_contextual_log_var, negative_log_var, emb_size), axis=0)
    # print(a.shape)
    # a = tf.reduce_mean(kld(word_contextual_mean, word_mean, word_contextual_log_var, word_log_var, emb_size), axis=0)
    # print(a.shape)

    loss = tf.maximum(0.,
        tf.reduce_mean(kld(word_contextual_mean, context_mean, word_contextual_log_var, context_log_var, emb_size), axis=0) - \
        tf.reduce_mean(kld(word_contextual_mean, negative_mean, word_contextual_log_var, negative_log_var, emb_size), axis=0)) + \
        tf.reduce_mean(kld(word_contextual_mean, word_mean, word_contextual_log_var, word_log_var, emb_size), axis=0) + \
        tf.reduce_sum(tf.exp(word_contextual_log_var))

    # print(loss.shape)

    train = tf.train.AdamOptimizer().minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return {'train':train,
            'init':init,
            'word_contextual_mean':word_contextual_mean,
            'loss':loss,
            'wids':word_ids,
            'saver':saver,
            'emb':emb_mean}


# model()


# ph = tf.placeholder(shape=(2,1), dtype=tf.int32)
#
# var = tf.Variable(val, dtype=tf.float32)
#
# slice = tf.nn.embedding_lookup(var,ph)
#
# print(np.array([1,2]).reshape(1,-1))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     e = sess.run(slice, {ph:np.array([1,2]).reshape(-1,1)})
#     print(e, e.shape)
