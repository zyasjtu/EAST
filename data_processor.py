import numpy as np
import cv2
import os
import math
import random

IMG_SIZE = 512
SRK_RATE = 0.1


def read_lines(path):
    with open(path) as f:
        return f.readlines()


def read_ant(ant_path):
    quad_list = []
    with open(ant_path) as f:
        lines = f.readlines()
        for line in lines:
            quad = line.rstrip('\n').split(',')
            quad_list.append(quad)
    return np.array(quad_list).astype('int')


def resize_with_padding(img, points, output_width, output_height):
    div = 1.0 * output_width / output_height
    input_height, input_width, _ = img.shape
    scale = 1.0
    if input_width == div * input_height:
        img = cv2.resize(img, (int(output_width), int(output_height)))
    elif input_width > div * input_height:
        padding = int((input_width / div - input_height) / 2)
        points[0][1] = points[0][1] + padding
        points[1][1] = points[1][1] + padding
        points[2][1] = points[2][1] + padding
        points[3][1] = points[3][1] + padding
        scale = 1.0 * input_width / output_width
        img = cv2.copyMakeBorder(img, padding, int(input_width / div - input_height - padding), 0, 0,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padding = int((div * input_height - input_width) / 2)
        points[0][0] = points[0][0] + padding
        points[1][0] = points[1][0] + padding
        points[2][0] = points[2][0] + padding
        points[3][0] = points[3][0] + padding
        scale = 1.0 * input_height / output_height
        img = cv2.copyMakeBorder(img, 0, 0, padding, int(input_height * div - input_width - padding),
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (output_width, output_height))
    points = np.array(points) / scale
    return img, points.astype('int')


def reorder_pts(xy_list, epsilon=1e-4):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
               / (xy_list[index, 0] - xy_list[first_v, 0] + epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
        xy_list[second_v, 0] - xy_list[fourth_v, 0] + epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=0.1):
    if ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])


def shrink_pts(xy_list, ratio=0.2, epsilon=1e-4):
    if ratio == 0.0:
        return xy_list, xy_list
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def pt_in_quad(px, py, quad, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad[1:4, :] - quad[:3, :]
        xy_list[3] = quad[0, :] - quad[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def generate_gt(gt, pts, shrinked_pts):
    rect = cv2.boxPoints(cv2.minAreaRect(pts))
    theta = math.atan2(rect[0][1] - rect[3][1], rect[3][0] - rect[0][0])
    idx = 1
    if theta >= np.pi / 4:
        theta = -(np.pi / 2 - theta)
        idx = 2

    p_min = np.amin(shrinked_pts, axis=0)
    p_max = np.amax(shrinked_pts, axis=0)
    for px in range(p_min[0], p_max[0]):
        for py in range(p_min[1], p_max[1]):
            if pt_in_quad(px, py, shrinked_pts, p_min, p_max):
                pt = np.array([px, py])
                gt[py, px, 0] = 1
                gt[py, px, 1] = dist_to_line(rect[idx], rect[idx + 1], pt)
                gt[py, px, 2] = dist_to_line(rect[idx + 1], rect[(idx + 2) % 4], pt)
                gt[py, px, 3] = dist_to_line(rect[(idx + 2) % 4], rect[(idx + 3) % 4], pt)
                gt[py, px, 4] = dist_to_line(rect[(idx + 3) % 4], rect[(idx + 4) % 4], pt)
                gt[py, px, 5] = theta


def write_lines(path, lines):
    with open(path, mode='w') as f:
        for line in lines:
            f.writelines(line + '\n')


def split_data(img_dir, ant_dir):
    fn_list = os.listdir(img_dir)
    for idx, fn in enumerate(fn_list):
        img = cv2.imread(os.path.join(img_dir, fn))
        quads = read_ant(os.path.join(ant_dir, fn[:len(fn) - 3] + 'txt'))

        pts_list = []
        for quad in quads:
            dst, pts = resize_with_padding(img, quad.reshape(4, 2), IMG_SIZE, IMG_SIZE)
            pts_list.append(pts)

        reordered_list = []
        for pts in pts_list:
            reordered_list.append(reorder_pts(pts).astype('float64'))

        shrinked_list = []
        for pts in reordered_list:
            _, shrinked, _ = shrink_pts(pts)
            shrinked_list.append(shrinked.astype('int'))

        gt = np.zeros([IMG_SIZE, IMG_SIZE, 6])
        for i in range(len(shrinked_list)):
            generate_gt(gt, pts_list[i], shrinked_list[i])

        cv2.imwrite(os.path.join(data_dir, "input_img", fn), dst)
        np.save(os.path.join(data_dir, "input_gt", fn[:len(fn) - 3] + 'npy'), gt[::4, ::4, ::])
        print(idx, len(fn_list), fn)

    random.shuffle(fn_list)
    valid_list = fn_list[:int(split_rate * len(fn_list))]
    train_list = fn_list[int(split_rate * len(fn_list)):]
    write_lines(os.path.join(data_dir, "train.list"), train_list)
    write_lines(os.path.join(data_dir, "valid.list"), valid_list)


def next_batch(img_dir, gt_dir, data_set, batch_size, start_idx=None):
    if start_idx is None:
        batch = random.sample(data_set, batch_size)
    else:
        batch = data_set[start_idx:start_idx + batch_size]

    img_list = []
    gt_list = []
    for line in batch:
        fn = line.rstrip('\n')
        img_list.append(cv2.imread(os.path.join(img_dir, fn)))
        gt_list.append(np.load(os.path.join(gt_dir, fn[:len(fn) - 3] + 'npy')))

    return np.array(img_list).astype('float32'), np.array(gt_list)


def restore_rectangle(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


split_rate = 0.1
data_dir = './data/'

if __name__ == '__main__':
    split_data(os.path.join(data_dir, "img"), os.path.join(data_dir, "gt"))
