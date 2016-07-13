
import os
import numpy as np
import scipy.io as spio
import scipy.signal as signal
import h5py

import time


_floatX = np.float32
_intX = np.int32


def load_mat_data(raw_data_dir, target, data_type):
    """
    :param raw_data_dir: raw data directory
    :param target: target is Dog_1, Dog_2, and so on.
    :param data_type: data_type is interictal, preictal and test
    :return: (segment, segment_name)
    """
    current_dir = os.path.join(raw_data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s\%s_%s_segment_%0.4d.mat' % (current_dir, target, data_type, i)
        segment_name = '%s_segment_%d' % (data_type, i)
        if os.path.exists(filename):
            mat_data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
            segment = mat_data[segment_name]
            yield (segment, segment_name)
        else:
            if 1 == i:
                raise Exception("file %s not found" % filename)
            done = True


def parse_mat_data(raw_data_dir, hdf_data_dir, target, data_type, resample_frequency=None):
    """
    :param raw_data_dir: raw data directory.
    :param hdf_data_dir: resample hdf data directory.
    :param target: target is Dog_1, Dog_2, and so on.
    :param data_type: data_type is interictal, preictal and test
    :param resample_frequency: target frequency for resampling
    :return: None
    """
    # pass
    _postfix = target + "_" + data_type + ".hdf"
    hdf_dir = os.path.join(hdf_data_dir, _postfix)
    f = h5py.File(hdf_dir, 'w', libver='latest')

    x = load_mat_data(raw_data_dir, target, data_type)

    for segment, segment_name in x:
        print('target: %s, segment_name: %s, ' % (target, segment_name), end='')
        grp = f.create_group(segment_name)
        grp.attrs['sampling_frequency'] = segment.sampling_frequency
        grp.attrs['data_length_sec'] = segment.data_length_sec
        if data_type is not 'test':
            grp.attrs['sequence'] = segment.sequence
        data = segment.data

        if resample_frequency is not None:
            if segment.sampling_frequency > resample_frequency:
                data = signal.resample(data, resample_frequency * segment.data_length_sec, axis=segment.data.ndim - 1)
                data = np.asarray(data, dtype=_floatX)
        print('shape:',  data.shape)
        grp.create_dataset("data", data=data, compression="gzip")
        channels = np.string_(list(segment.channels))
        grp.create_dataset("channels", data=channels)


def mat_to_hdf(raw_data_dir, hdf_data_dir, resample_frequency=None):
    """
    transform .mat data files into .hdf data files for all targets and all data types,
    every possible pair of target and data type corresponds to a separate .hdf file.
    :return: None
    """
    targets = ['Patient_1', 'Patient_2', 'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']
    data_types = ['preictal', 'interictal', 'test']

    for target in targets:
        for data_type in data_types:
            start_time = time.time()
            print('processing raw data for %s of data type %s' % (target, data_type))
            parse_mat_data(raw_data_dir, hdf_data_dir, target, data_type, resample_frequency)
            end_time = time.time()
            print('processing raw data for %s of data type %s completed, took time %.4f' %
                  (target, data_type, end_time - start_time))


def load_hdf_data(hdf_data_dir, target, data_type, verbose=False):
    _postfix = target + "_" + data_type + ".hdf"
    hdf_dir = os.path.join(hdf_data_dir, _postfix)

    f = h5py.File(hdf_dir, 'r',  libver='latest')
    data = list()
    for key in f.keys():
        data.append(f[key]['data'])
    data = np.asarray(data, dtype=_floatX)
    if verbose:
        print('loading data from hdf files, target: %s, data type: %s, data shape: %s' % (target, data_type, data.shape))

    return data


def split_stream(data, n_window, window_length, overlapping_length=0):
    """
    :param data: data stream shape: [n_channel, n_feature]
    :param n_window: window number
    :param window_length: window length
    :param overlapping_length: overlapping length
    :return: split data (np.array) with shape [n_window, n_channel, window_length]
    """

    n_channel, n_feature = data.shape

    needed_n_feature = (n_window - 1) * (window_length - overlapping_length) + window_length

    if n_feature >= needed_n_feature:
        pass
    else:
        raise ValueError('feature length: %d, needed feature length: %d' % (n_feature, needed_n_feature))

    if window_length > overlapping_length:
        pass
    else:
        raise ValueError('window length %f must be larger than overlapping length %f' % (window_length, overlapping_length))

    starting_points = []
    for i in range(n_window):
        idx = max(i * (window_length - overlapping_length), 0)
        if idx >= n_feature:
            print('split error, start idx: %d, n_feature: %d' % (idx, n_feature))
            break
        elif idx + window_length > n_feature:
            print('split error, end idx: %d, n_feature: %d' % (idx+window_length, n_feature))
            break
        else:
            pass
        starting_points.append(idx)

    split_data = []
    for idx in starting_points:
        split_data.append(data[:, idx:idx + window_length])

    split_data = np.asarray(split_data, dtype=_floatX)

    return split_data


def windower(data, n_window, window_length, overlapping_length=0):
    """
    :param data: raw data with shape [n_segment, n_channel, n_feature]
    :param n_window:  window number
    :param window_length:  window length
    :param overlapping_length:  overlapping length
    :return: data with shape [n_sample, n_channel, n_feature] and label with shape [n_sample]
    """

    windowed_data = []
    for segment in data:
        windowed_data.extend(split_stream(segment, n_window, window_length, overlapping_length))

    windowed_data = np.asarray(windowed_data, dtype=_floatX)

    return windowed_data


def load_and_window_data_instance(hdf_data_dir, target, data_type, n_window, window_length, overlapping_length):

    print('instance mode, split data starting, n_window: %d, window_length: %s, overlapping_length: %d' % (
        n_window, window_length, overlapping_length))

    data = load_hdf_data(hdf_data_dir, target, data_type)
    window_data = windower(data, n_window, window_length, overlapping_length)

    if data_type == 'preictal':
        label = np.ones(shape=[window_data.shape[0], ], dtype=np.int8)
    elif data_type == 'interictal':
        label = np.zeros(shape=[window_data.shape[0], ], dtype=np.int8)
    else:
        label = None

    print('split data completed, target: %s, data type: %s, windowed data shape: %s, label shape: %s' % (
        target, data_type, window_data.shape, label.shape))

    return window_data, label


def load_and_window_data_bag(hdf_data_dir, target, data_type, n_window, window_length, overlapping_length):
    print('bag mode, split data starting, n_window: %d, window_length: %s, overlapping_length: %d' % (
        n_window, window_length, overlapping_length))
    data = load_hdf_data(hdf_data_dir, target, data_type)
    window_data = windower(data, n_window, window_length, overlapping_length)

    # print(window_data.shape)
    n_instances, n_channel, n_feature = window_data.shape
    n_segment = int(n_instances / n_window)
    # print(n_segment)
    window_data = window_data.reshape([n_segment, n_window, n_channel, n_feature])
    # print(window_data.shape)

    if data_type == 'preictal':
        label = np.ones(shape=[window_data.shape[0], ], dtype=np.int8)
    elif data_type == 'interictal':
        label = np.zeros(shape=[window_data.shape[0], ], dtype=np.int8)
    else:
        label = None

    print('split data completed, target: %s, data type: %s, windowed data shape: %s, label shape: %s' % (
        target, data_type, window_data.shape, label.shape))

    return window_data, label


if __name__ == '__main__':
    from setting import load_settings
    settings = load_settings()
    raw_data_dir = settings.raw_data_dir
    hdf_data_dir = settings.hdf_data_dir
    resample_hdf_data_dir = settings.resample_hdf_data_dir

    target = 'Patient_1'
    data_type = 'preictal'
    # data_type = 'interictal'
    # mat_to_hdf(raw_data_dir, hdf_data_dir)

    # mat_to_hdf(raw_data_dir, resample_hdf_data_dir, target_frequency=100)

    # data, label = load_hdf_data(resample_hdf_data_dir, target, data_type)
    #
    # windower(data, n_window=12, window_length=5000, overlapping_length=0)

    # load_and_window_data_instance(resample_hdf_data_dir, target, data_type, n_window=12, window_length=5000, overlapping_length=0)
    load_and_window_data_bag(resample_hdf_data_dir, target, data_type, n_window=12, window_length=5000, overlapping_length=0)