from sklearn import svm
import numpy as np
import time
from sklearn import cross_validation
import random


from setting import load_settings
from data_utils import load_and_window_data_instance, load_and_window_data_bag
from time_series_features import EegSignalFeature

_floatX = np.float32
_intX = np.int8


class SVM(object):

    def __init__(self):
        self.eeg_feature_utils = EegSignalFeature()
        # pass

    def generate_feature_and_label_instance(self, hdf_data_dir, target, data_type, start_hz, end_hz, option, resample_size, max_hz):

        data, label = load_and_window_data_instance(hdf_data_dir, target, data_type, n_window=12, window_length=500, overlapping_length=0)
        feature_set = list()
        for instance in data:
            feature = self.eeg_feature_utils.fft_with_time_frequency_correlation(instance, start_hz, end_hz, option, resample_size, max_hz)
            feature_set.append(feature)
        feature_set = np.asarray(feature_set, dtype=_floatX)
        return feature_set, label

    def generate_feature_and_label_bag(self, hdf_data_dir, target, data_type, start_hz, end_hz, option, resample_size, max_hz):
        data, label = load_and_window_data_bag(hdf_data_dir, target, data_type, n_window=12, window_length=500,
                                               overlapping_length=0)
        print(data.shape)
        print(label.shape)
        feature_set = list()
        n_segment, n_window, n_channel, n_data = data.shape
        for seg_idx in range(n_segment):
            for win_idx in range(n_window):
                feature = self.eeg_feature_utils.fft_with_time_frequency_correlation(data[seg_idx, win_idx, :, :], start_hz, end_hz, option, resample_size, max_hz)
                feature_set.append(feature)
        feature_set = np.asarray(feature_set, dtype=_floatX)
        print(feature_set.shape)
        n_instance, n_feature = feature_set.shape
        feature_set = feature_set.reshape([n_segment, n_window, n_feature])
        print(feature_set.shape)
        print(label.shape)

        return feature_set, label

    def flatten_feature_and_label(self, feature, label):
        # n_dim = feature.ndim
        n_segment, n_window, n_feature = feature.shape
        # print(n_segment, n_window, n_feature)
        feature = feature.reshape([-1, n_feature])
        label = label.reshape([label.shape[0], 1])
        # print(label.shape)
        label = np.tile(label, [1, n_window])
        # print(feature.shape)
        # print(label.shape)
        # print(label)
        label = label.ravel()
        # print(label.shape)
        # print(label)
        # i = 0
        # while i < len(label):
        #     print(label[i:i+12])
        #     i = i + 12
        return feature, label


    def train_test_split(self, feature, label, split_ratio):
        random_seed = random.randint(1, 1000)
        # random_seed = 10
        train_x, test_x, train_y, test_y = cross_validation.train_test_split(feature, label, test_size=split_ratio, random_state=random_seed)

        train_data = (train_x, train_y)
        test_data = (test_x, test_y)

        return train_data, test_data

    def train(self, train_data):
        train_x, train_y = train_data
        # in original proposal, C is 1e-6, but C=1e6 achieve better performance
        clf = svm.SVC(kernel='rbf', C=1e6, gamma=0.01, coef0=0.0, shrinking=True)
        clf.fit(train_x, train_y)
        return clf

    def predict(self, clf, test_data):
        test_x, test_y = test_data
        predict_y = clf.predict(test_x)
        print('test y:', test_y)
        print('predict y:', predict_y)
        print(sum(test_y == predict_y) / len(test_y))


def test_instance():
    settings = load_settings()
    raw_data_dir = settings.raw_data_dir
    hdf_data_dir = settings.hdf_data_dir
    resample_hdf_data_dir = settings.resample_hdf_data_dir
    svm_obj = SVM()

    start_hz = 1
    end_hz = 51
    max_hz = None
    resample_size = 18
    option = 'usr'

    target = 'Dog_1'
    data_type = 'preictal'

    preictal_feature, preictal_label = svm_obj.generate_feature_and_label_instance(resample_hdf_data_dir, target,
                                                                                   data_type,
                                                                                   start_hz, end_hz, option,
                                                                                   resample_size, max_hz)

    data_type = 'interictal'
    interictal_feature, interictal_label = svm_obj.generate_feature_and_label_instance(resample_hdf_data_dir, target,
                                                                                       data_type,
                                                                                       start_hz, end_hz, option,
                                                                                       resample_size, max_hz)

    feature = np.concatenate([preictal_feature, interictal_feature], axis=0)
    label = np.concatenate([preictal_label, interictal_label], axis=0)
    print(feature.shape, label.shape)

    train_data, test_data = svm_obj.train_test_split(feature, label, split_ratio=0.2)
    train_x, train_y = train_data
    test_x, test_y = test_data
    print(train_x.shape)
    print(test_x.shape)
    # print(test_y)
    clf = svm_obj.train(train_data)
    svm_obj.predict(clf, test_data)

def test_bag():
    settings = load_settings()
    raw_data_dir = settings.raw_data_dir
    hdf_data_dir = settings.hdf_data_dir
    resample_hdf_data_dir = settings.resample_hdf_data_dir
    svm_obj = SVM()

    # start_hz = 50
    # end_hz = 2500
    start_hz = 1
    end_hz = 51
    max_hz = None
    resample_size = 18
    option = 'usr'

    target = 'Patient_1'
    data_type = 'preictal'

    preictal_feature, preictal_label = svm_obj.generate_feature_and_label_bag(resample_hdf_data_dir, target, data_type, start_hz, end_hz, option, resample_size, max_hz)

    data_type = 'interictal'
    interictal_feature, interictal_label = svm_obj.generate_feature_and_label_bag(resample_hdf_data_dir, target, data_type,
                                                                              start_hz, end_hz, option, resample_size,
                                                                              max_hz)

    feature = np.concatenate([preictal_feature, interictal_feature], axis=0)
    label = np.concatenate([preictal_label, interictal_label], axis=0)
    # print(feature.shape, label.shape)

    train_data, test_data = svm_obj.train_test_split(feature, label, split_ratio=0.2)
    train_x, train_y = train_data
    test_x, test_y = test_data
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    # print(train_y)
    # print(test_y)
    train_x, train_y = svm_obj.flatten_feature_and_label(train_x, train_y)
    test_x, test_y = svm_obj.flatten_feature_and_label(test_x, test_y)
    train_data = (train_x, train_y)
    test_data = (test_x, test_y)
    clf = svm_obj.train(train_data)
    svm_obj.predict(clf, test_data)


if __name__ == '__main__':

    # test_instance()
    test_bag()
