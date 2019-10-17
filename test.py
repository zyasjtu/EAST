import tensorflow as tf
import east_model
import data_processor
import cv2
import os
import numpy as np
import time
import east_utils

IMG_SIZE = 512


def post_process(score_map, geo_map, timer, score_map_thresh=0.9, box_thresh=0.1, nms_thres=0.2):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = data_processor.restore_rectangle(xy_text[:, ::-1] * 4,
                                                         geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = east_utils.la_nms(boxes.astype(np.float64), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def test(img_dir, test_list):
    data_list = data_processor.read_lines(test_list)

    img_input = tf.placeholder(tf.float32, shape=[1, IMG_SIZE, IMG_SIZE, 3])

    network = east_model.EAST(training=False, max_len=IMG_SIZE)
    pred_score, pred_gmt = network.build(img_input)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        tf.train.Saver().restore(sess=sess, save_path=tf.train.latest_checkpoint('./checkpoint'))
        for fn in data_list:
            fn = fn.rstrip('\n').rstrip(',')
            print(os.path.join(img_dir, fn))
            bgr = cv2.imread(os.path.join(img_dir, fn)).astype(np.float32)

            start_time = time.time()
            timer = {'net': 0, 'restore': 0, 'nms': 0}
            bgr_input, _ = data_processor.resize_with_padding(bgr, np.zeros([4, 2]), IMG_SIZE, IMG_SIZE)
            start = time.time()
            s, g = sess.run([pred_score, pred_gmt], feed_dict={img_input: [bgr_input]})
            timer['net'] = time.time() - start
            boxes, timer = post_process(s, g, timer)
            print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(fn, timer['net'] * 1000,
                                                                             timer['restore'] * 1000,
                                                                             timer['nms'] * 1000))
            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
            duration = time.time() - start_time
            print('[timing] {}'.format(duration))
            if (boxes is not None) and (len(boxes[0]) > 0):
                print(boxes)
                cv2.polylines(bgr_input, np.int32(boxes), True, (0, 0, 255))
                cv2.imwrite('./demo/' + fn[:len(fn) - 4] + '_img.jpg', bgr_input)


if __name__ == '__main__':
    test('./data/test', './data/test.list')
