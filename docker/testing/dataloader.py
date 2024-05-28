import pandas as pd
import cv2
from matplotlib import pyplot as plt
from torch.utils import data
from torch.utils.data import DataLoader
from dependency import *
import torch
import numpy as np
from utils import encode_label, encode_meta_label
# from keras.utils import to_categorical
import albumentations
# Build the Pytorch dataloader
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Resize,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    ShiftScaleRotate,
    RandomBrightnessContrast,
)


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


aug = Compose(
    [
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5,
                         rotate_limit=45, p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.5),
        CenterCrop(224, 224)
        # Resize(224, 224)
        # RandomContrast(p=0.5),
        # RandomBrightness(p=0.5),
        # RandomGamma(p=0.5)

    ],
    p=1)


def load_image(path, shape):
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (shape[0], shape[1]))
    except Exception as e:
        print(str(e))

    return img


class SkinDataset(data.Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_test=False, num_class=1):
        self.is_test = is_test
        self.image_dir = image_dir
        self.img_info = img_info
        self.file_list = file_list
        self.shape = shape
        self.num_class = num_class
        self.total_img_info = img_info
        
        # print(self.total_img_info)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))


        file_id = self.file_list[index]
        sub_img_info = self.total_img_info[file_id:file_id + 1]

        # print(sub_img_info)

        # get the clincal image path
        clinic_img_path = sub_img_info['clinic']
        # get the dermoscopy image path
        dermoscopy_img_path = sub_img_info['derm']
        # load the clinical image
        clinic_img = load_image(
            self.image_dir + clinic_img_path[file_id], self.shape)
        # load the dermoscopy image
        dermoscopy_img = load_image(
            self.image_dir + dermoscopy_img_path[file_id], self.shape)

        # Encode the diagnositic label
        diagnosis_label = sub_img_info['diagnosis'][file_id]
        for index_label, label in enumerate(label_list):
            if diagnosis_label in label:
                diagnosis_index = index_label
                diagnosis_label_one_hot = to_categorical(
                    diagnosis_index, num_label)
            else:
                continue

        # if not self.is_test:
        # print(sub_img_info)
        augmented = aug(image=clinic_img, mask=dermoscopy_img)
        clinic_img = augmented['image']
        dermoscopy_img = augmented['mask']

        # print(file_id, sub_img_info)

        total_label = encode_label(sub_img_info, file_id)
        # print(total_label)
        clinic_img = torch.from_numpy(np.transpose(
            clinic_img, (2, 0, 1)).astype('float32') / 255)
        dermoscopy_img = torch.from_numpy(np.transpose(
            dermoscopy_img, (2, 0, 1)).astype('float32') / 255)
        # meta_data = encode_meta_label(sub_img_info, file_id)

        # print(file_id, total_label[7])
        return clinic_img, dermoscopy_img, [total_label[0], total_label[1], total_label[2], total_label[3],
                                            total_label[4], total_label[5], total_label[6], total_label[7]]


def demo_test():
    # train,val,test dataset spliting
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]

    df = pd.read_csv(img_info_path)

    # Img Information
    index_num = 7
    img_info = df[index_num:index_num+1]
    clinic_path = img_info['clinic']
    dermoscopy_path = img_info['derm']
    # source_dir = '../release_v0/release_v0/images/'
    clinic_img = cv2.imread(source_dir+clinic_path[index_num])
    dermoscopy_img = cv2.imread(source_dir+dermoscopy_path[index_num])

    plt.subplot(121)
    plt.imshow(dermoscopy_img)

    plt.subplot(122)
    plt.imshow(clinic_img)
    plt.show()


def demo_run():
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    df = pd.read_csv(img_info_path)

    shape = (224, 224)
    batch_size = 16
    num_workers = 0
    train_skindataset = SkinDataset(image_dir=source_dir,
                                    img_info=df,
                                    file_list=train_index_list,
                                    shape=shape, is_test=False)

    train_dataloader = DataLoader(
        dataset=train_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True)

    val_skindataset = SkinDataset(image_dir=source_dir,
                                  img_info=df,
                                  file_list=val_index_list,
                                  shape=shape, is_test=True)

    val_dataloader = DataLoader(
        dataset=val_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True)

    for clinic_img, derm_img, label in train_dataloader:
        print(clinic_img.shape, derm_img.shape, label[0].shape)
        print('train_dataloader finished')

    for clinic_img, derm_img, label in val_dataloader:
        print(clinic_img.shape, derm_img.shape, label[0].shape)
        print('val_dataloader finished')


def generate_dataloader(shape, batch_size, num_workers, data_mode):
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]

    df = pd.read_csv(img_info_path)
    if data_mode == 'self_evaluated':
        data_mode = 'SP'
        print('--------------------------', source_dir)
        train_skindataset = SkinDataset(image_dir=source_dir,
                                        img_info=df,
                                        file_list=train_index_list_1,
                                        shape=shape, is_test=False,
                                        )
        train_dataloader = DataLoader(
            dataset=train_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

        val_skindataset = SkinDataset(image_dir=source_dir,
                                      img_info=df,
                                      file_list=val_index_list,
                                      shape=shape, is_test=True,
                                      )

        val_dataloader = DataLoader(
            dataset=val_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

    else:
        # print('--------------------------', source_dir)

        train_skindataset = SkinDataset(image_dir=source_dir,
                                        img_info=df,
                                        file_list=train_index_list,
                                        shape=shape,
                                        is_test=False,
                                        )
        train_dataloader = DataLoader(
            dataset=train_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True)

        val_skindataset = SkinDataset(image_dir=source_dir,
                                      img_info=df,
                                      file_list=val_index_list,
                                      shape=shape,
                                      is_test=True,
                                      )

        val_dataloader = DataLoader(
            dataset=val_skindataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False)
    print('dataloader created!')
    return train_dataloader, val_dataloader
