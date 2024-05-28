from dependency import *
# from keras.utils import to_categorical
import numpy as np
import os
import torch

# def to_categorical(
# def to_categorical(targets, nb_classes):
#    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
#    return res.reshape(list(targets.shape)+[nb_classes])


def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    # return {'Total': total_num, 'Trainable': trainable_num}


def create_cosine_learing_schdule(epochs, lr):
    cosine_learning_schule = []

    for epoch in range(epochs):
        # t - 1 is used when t has 1-based indexing.
        cos_inner = np.pi * (epoch % epochs)
        cos_inner /= epochs
        cos_out = np.cos(cos_inner) + 1
        final_lr = float(lr / 2 * cos_out)
        cosine_learning_schule.append(final_lr)

    return cosine_learning_schule


class Logger:

    def __int__(self):
        super(Logger, self).__int__()

    def open(self, name, mode):
        self.txt = open(name, mode=mode)

    def write(self, str_):
        self.txt.write(str_)
        print(str_)

    def close(self):
        self.txt.close()


def set_seed(seed=15):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def CraateLogger(mode, model_name='resnet-50', round_=None, data_mode='Normal'):

    out_dir = './{}_{}_{}_weight_file/{}/'.format(
        mode, model_name, data_mode, round_)
    os.makedirs(out_dir + '/checkpoint/', exist_ok=True)
    os.makedirs(out_dir + '/train/', exist_ok=True)
    # os.makedirs(out_dir + '/backup/', exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.single_modality_{}_skinlesion.txt'.format(mode), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    # log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments> ...  \n')
    log.write('\t  - {} {}  \n'.format(mode, model_name))
    log.write('\t  - simple augmentation \n')
    log.write('\n')

    return log, out_dir


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# label_list = [nevus_list, basal_cell_carcinoma_list, melanoma_list, miscellaneous_list, SK_list]

def encode_label(img_info, index_num):
    # Encode the diagnositic label
    diagnosis_label = img_info['diagnosis'][index_num]
    if diagnosis_label in label_list[2]:
        diagnosis_index = 1
    else:
        diagnosis_index = 0
    # for index, label in enumerate(label_list):
        # if diagnosis_label in label:
        #     diagnosis_index = index
        #     diagnosis_label_one_hot = to_categorical(
        #         diagnosis_index, num_label)
        # else:
        #     continue
    # Encode the Seven-point label
    # 1
    pigment_network_label = img_info['pigment_network'][index_num]
    pigment_network_index = pigment_network_label.item()
    # for index, label in enumerate(pigment_network_label_list):
    #     if pigment_network_label in label:
    #         pigment_network_index = index
    #         print(type(pigment_network_index))
    #         pigment_network_label_one_hot = to_categorical(
    #             pigment_network_index, num_pigment_network_label)
    #     else:
    #         continue
    # 2
    streaks_label = img_info['streaks'][index_num]
    streaks_index = streaks_label.item()
    # for index, label in enumerate(streaks_label_list):
    #     if streaks_label in label:
    #         streaks_index = index
    #         streaks_label_one_hot = to_categorical(
    #             streaks_index, num_streaks_label)
    #     else:
    #         continue
    # 3
    pigmentation_label = img_info['pigmentation'][index_num]
    pigmentation_index = pigmentation_label.item()
    # for index, label in enumerate(pigmentation_label_list):
    #     if pigmentation_label in label:
    #         pigmentation_index = index
    #         pigmentation_label_one_hot = to_categorical(
    #             pigmentation_index, num_pigmentation_label)
    #     else:
    #         continue
    # 4
    regression_structures_label = img_info['regression_structures'][index_num]
    regression_structures_index = regression_structures_label.item()
    # for index, label in enumerate(regression_structures_label_list):
    #     if regression_structures_label in label:
    #         regression_structures_index = index
    #         regression_structures_label_one_hot = to_categorical(regression_structures_index,
    #                                                              num_regression_structures_label)
    #     else:
    #         continue
    # 5
    dots_and_globules_label = img_info['dots_and_globules'][index_num]
    dots_and_globules_index = dots_and_globules_label.item()
    # for index, label in enumerate(dots_and_globules_label_list):
    #     if dots_and_globules_label in label:
    #         dots_and_globules_index = index
    #         dots_and_globules_label_one_hot = to_categorical(
    #             dots_and_globules_index, num_dots_and_globules_label)
    #     else:
    #         continue
    # 6
    blue_whitish_veil_label = img_info['blue_whitish_veil'][index_num]
    blue_whitish_veil_index = blue_whitish_veil_label.item()
    # for index, label in enumerate(blue_whitish_veil_label_list):
    #     if blue_whitish_veil_label in label:
    #         blue_whitish_veil_index = index
    #         blue_whitish_veil_label_one_hot = to_categorical(
    #             blue_whitish_veil_index, num_blue_whitish_veil_label)
    #     else:
    #         continue
    # 7
    vascular_structures_label = img_info['vascular_structures'][index_num]
    vascular_structures_index = vascular_structures_label.item()
    # for index, label in enumerate(vascular_structures_label_list):
    #     if vascular_structures_label in label:
    #         vascular_structures_index = index
    #         vascular_structures_label_one_hot = to_categorical(
    #             vascular_structures_index, num_vascular_structures_label)
    #     else:
    #         continue

    return np.array([diagnosis_index,
                     pigment_network_index,
                     streaks_index,
                     pigmentation_index,
                     regression_structures_index,
                     dots_and_globules_index,
                     blue_whitish_veil_index,
                     vascular_structures_index])


def encode_test_label(img_info, index_num):
    # Encode the diagnositic label
    diagnosis_label = img_info['diagnosis'][index_num]
    for index, label in enumerate(label_list):
        if diagnosis_label in label:
            diagnosis_index = index
            diagnosis_label_one_hot = to_categorical(
                diagnosis_index, num_label)
        # print(index_num,diagnosis_index,diagnosis_label,diagnosis_label_one_hot)
        else:
            continue

    # Encode the Seven-point label
    # 1
    pigment_network_label = img_info['pigment_network'][index_num]
    for index, label in enumerate(pigment_network_label_list):
        if pigment_network_label in label:
            pigment_network_index = index
            pigment_network_label_one_hot = to_categorical(
                pigment_network_index, num_pigment_network_label)
        else:
            continue
    # 2
    streaks_label = img_info['streaks'][index_num]
    for index, label in enumerate(streaks_label_list):
        if streaks_label in label:
            streaks_index = index
            streaks_label_one_hot = to_categorical(
                streaks_index, num_streaks_label)
        else:
            continue
    # 3
    pigmentation_label = img_info['pigmentation'][index_num]
    for index, label in enumerate(pigmentation_label_list):
        if pigmentation_label in label:
            pigmentation_index = index
            pigmentation_label_one_hot = to_categorical(
                pigmentation_index, num_pigmentation_label)
        else:
            continue
    # 4
    regression_structures_label = img_info['regression_structures'][index_num]
    for index, label in enumerate(regression_structures_label_list):
        if regression_structures_label in label:
            regression_structures_index = index
            regression_structures_label_one_hot = to_categorical(regression_structures_index,
                                                                 num_regression_structures_label)
        else:
            continue
    # 5
    dots_and_globules_label = img_info['dots_and_globules'][index_num]
    for index, label in enumerate(dots_and_globules_label_list):
        if dots_and_globules_label in label:
            dots_and_globules_index = index
            dots_and_globules_label_one_hot = to_categorical(
                dots_and_globules_index, num_dots_and_globules_label)
        else:
            continue
    # 6
    blue_whitish_veil_label = img_info['blue_whitish_veil'][index_num]
    for index, label in enumerate(blue_whitish_veil_label_list):
        if blue_whitish_veil_label in label:
            blue_whitish_veil_index = index
            blue_whitish_veil_label_one_hot = to_categorical(
                blue_whitish_veil_index, num_blue_whitish_veil_label)
        else:
            continue
    # 7
    vascular_structures_label = img_info['vascular_structures'][index_num]
    for index, label in enumerate(vascular_structures_label_list):
        if vascular_structures_label in label:
            vascular_structures_index = index
            vascular_structures_label_one_hot = to_categorical(
                vascular_structures_index, num_vascular_structures_label)
        else:
            continue

    return np.array([diagnosis_index,
                     pigment_network_index,
                     streaks_index,
                     pigmentation_index,
                     regression_structures_index,
                     dots_and_globules_index,
                     blue_whitish_veil_index,
                     vascular_structures_index]), np.array([diagnosis_label_one_hot,
                                                            pigment_network_label_one_hot,
                                                            streaks_label_one_hot,
                                                            pigmentation_label_one_hot,
                                                            regression_structures_label_one_hot,
                                                            dots_and_globules_label_one_hot,
                                                            blue_whitish_veil_label_one_hot,
                                                            vascular_structures_label_one_hot])


def encode_meta_label(img_info, index_num):

    level_of_diagnostic_difficulty_label = img_info['level_of_diagnostic_difficulty'][index_num]
    # print(level_of_diagnostic_difficulty_label)
    for index, label in enumerate(level_of_diagnostic_difficulty_label_list):

        if level_of_diagnostic_difficulty_label in label:
            level_of_diagnostic_difficulty_index = index
            level_of_diagnostic_difficulty_label_one_hot = to_categorical(
                level_of_diagnostic_difficulty_index, num_level_of_diagnostic_difficulty_label_list)
        else:
            continue

    evaluation_label = img_info['elevation'][index_num]
    for index, label in enumerate(evaluation_list):
        if evaluation_label in label:
            evaluation_label_index = index
            evaluation_label_one_hot = to_categorical(
                evaluation_label_index, num_evaluation_list)
        else:
            continue

    sex_label = img_info['sex'][index_num]
    for index, label in enumerate(sex_list):
        if sex_label in label:
            sex_label_index = index
            sex_label_one_hot = to_categorical(sex_label_index, num_sex_list)
        else:
            continue

    location_label = img_info['location'][index_num]
    for index, label in enumerate(location_list):
        if location_label in label:
            location_label_index = index
            location_label_one_hot = to_categorical(
                location_label_index, num_location_list)
        else:
            continue

    management_label = img_info['management'][index_num]
    for index, label in enumerate(management_list):
        if management_label in label:
            management_label_index = index
            management_label_one_hot = to_categorical(
                management_label_index, num_management_list)
        else:
            continue

    meta_vector = np.hstack([
        level_of_diagnostic_difficulty_label_one_hot,
        evaluation_label_one_hot,
        location_label_one_hot,
        sex_label_one_hot,
        management_label_one_hot
    ])

    return meta_vector


def encode_meta_label_extra(img_info, index_num):

    level_of_diagnostic_difficulty_label = img_info['level_of_diagnostic_difficulty'][index_num]
    # print(level_of_diagnostic_difficulty_label)
    for index, label in enumerate(level_of_diagnostic_difficulty_label_list):

        if level_of_diagnostic_difficulty_label in label:
            level_of_diagnostic_difficulty_index = index
            level_of_diagnostic_difficulty_label_one_hot = to_categorical(
                level_of_diagnostic_difficulty_index, num_level_of_diagnostic_difficulty_label_list)
        else:
            continue

    evaluation_label = img_info['elevation'][index_num]
    for index, label in enumerate(evaluation_list):
        if evaluation_label in label:
            evaluation_label_index = index
            evaluation_label_one_hot = to_categorical(
                evaluation_label_index, num_evaluation_list)
        else:
            continue

    sex_label = img_info['sex'][index_num]
    for index, label in enumerate(sex_list):
        if sex_label in label:
            sex_label_index = index
            sex_label_one_hot = to_categorical(sex_label_index, num_sex_list)
        else:
            continue

    location_label = img_info['location'][index_num]
    for index, label in enumerate(location_list):
        if location_label in label:
            location_label_index = index
            location_label_one_hot = to_categorical(
                location_label_index, num_location_list)
        else:
            continue

    management_label = img_info['management'][index_num]
    for index, label in enumerate(management_list):
        if management_label in label:
            management_label_index = index
            management_label_one_hot = to_categorical(
                management_label_index, num_management_list)
        else:
            continue

    meta_vector = np.hstack([
        level_of_diagnostic_difficulty_label_one_hot,
        evaluation_label_one_hot,
        location_label_one_hot,
        sex_label_one_hot,
        management_label_one_hot
    ])

    return meta_vector
