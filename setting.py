from collections import namedtuple
import json
import os

Settings = namedtuple('Settings', ['raw_data_dir', 'hdf_data_dir', 'resample_hdf_data_dir'])


def load_settings():
    with open('SETTINGS.json') as f:
        settings = json.load(f)

    raw_data_dir = str(settings['raw-data-dir'])
    hdf_data_dir = str(settings['hdf-data-dir'])
    resample_hdf_data_dir = str(settings['resample-hdf-data-dir'])

    if not os.path.exists(raw_data_dir):
        raise NameError('raw data dir is not exist...')

    if os.path.exists(hdf_data_dir):
        # print('hdf data dir is already exist...')
        pass
    else:
        print('Initializing hdf data dir...')
        os.makedirs(hdf_data_dir)

    if os.path.exists(resample_hdf_data_dir):
        pass
    else:
        print('Initializing resample hdf data dir...')
        os.makedirs(resample_hdf_data_dir)

    return Settings(raw_data_dir=raw_data_dir, hdf_data_dir=hdf_data_dir, resample_hdf_data_dir=resample_hdf_data_dir)


if __name__ == '__main__':
    load_settings()
