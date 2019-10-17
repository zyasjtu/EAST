import tensorflow as tf
import east_model
import data_processor
import cv2

BATCH_SIZE = 16
IMG_SIZE = 512


def valid(img_dir, gt_dir, valid_list):
    data_list = data_processor.read_lines(valid_list)

    img_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    gt_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE // 4, IMG_SIZE // 4, 6])

    network = east_model.EAST(training=False, max_len=IMG_SIZE)
    pred_score, pred_gmt = network.build(img_input)
    loss = network.loss(gt_input[:, :, :, 0:1], pred_score, gt_input[:, :, :, 1:6], pred_gmt)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        tf.train.Saver().restore(sess=sess, save_path=tf.train.latest_checkpoint('./checkpoint'))
        n_step = len(data_list) // BATCH_SIZE
        for n in range(n_step):
            img_batch, gt_batch = data_processor.next_batch(img_dir, gt_dir, data_list, BATCH_SIZE, n * BATCH_SIZE)
            s, g, l = sess.run([pred_score, pred_gmt, loss], feed_dict={img_input: img_batch, gt_input: gt_batch})
            print(n, l)
            for m in range(BATCH_SIZE):
                img_show = cv2.resize(img_batch[m], (IMG_SIZE // 4, IMG_SIZE // 4))
                img_show[:, :, 0] += s[m][:, :, 0] * 255
                img_show[:, :, 1] += s[m][:, :, 0] * 255
                img_show[:, :, 2] += s[m][:, :, 0] * 255
                cv2.imwrite('./demo/' + str(n) + '_' + str(m) + '_img.jpg', img_show)
                cv2.imwrite('./demo/' + str(n) + '_' + str(m) + '_scr.jpg', s[m] * 255)


if __name__ == '__main__':
    valid('./data/input_img', './data/input_gt', './data/valid.list')
