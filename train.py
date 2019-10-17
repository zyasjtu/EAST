import tensorflow as tf
from tensorflow.contrib import slim
import east_model
import data_processor

BATCH_SIZE = 32
IMG_SIZE = 512
RESTORE = False


def train(img_dir, gt_dir, train_list, pretrained_path):
    img_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    gt_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE // 4, IMG_SIZE // 4, 6])

    network = east_model.EAST(training=True, max_len=IMG_SIZE)
    pred_score, pred_gmt = network.build(img_input)
    loss = network.loss(gt_input[:, :, :, 0:1], pred_score, gt_input[:, :, :, 1:6], pred_gmt)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(0.0001, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

    restore_op = slim.assign_from_checkpoint_fn(pretrained_path, slim.get_trainable_variables(),
                                                ignore_missing_vars=True)
    saver = tf.train.Saver()
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        if RESTORE:
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        else:
            sess.run(tf.global_variables_initializer())
            restore_op(sess)

        data_list = data_processor.read_lines(train_list)
        for step in range(100001):
            img_batch, gt_batch = data_processor.next_batch(img_dir, gt_dir, data_list, BATCH_SIZE)
            s, g, l, lr, _ = sess.run([pred_score, pred_gmt, loss, learning_rate, optimizer],
                                      feed_dict={img_input: img_batch, gt_input: gt_batch})
            if step % 1000 == 0 and step > 0:
                saver.save(sess=sess, save_path='./checkpoint/east.ckpt', global_step=step)
            if step % 100 == 0:
                print(step, lr, l)


if __name__ == '__main__':
    train('./data/input_img', './data/input_gt', './data/train.list', './pretrained/resnet_v1_50.ckpt')
