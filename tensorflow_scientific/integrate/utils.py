import tensorflow as tf


def flatten(seq):
    if not seq:
        return tf.convert_to_tensor([])
    flat = [tf.reshape(p, [-1]) for p in seq]
    return tf.concat(flat, 0)
