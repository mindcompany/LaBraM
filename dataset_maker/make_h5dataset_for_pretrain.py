import gc
import glob
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from dataset_maker.shock.utils import h5Dataset
from dataset_maker.shock.utils import preprocessing_cnt
from dataset_maker.shock.utils import preprocessing_edf

# savePath = Path('path/to/your/save/path')
# rawDataPath = Path('path/to/your/raw/data/path')
# group = rawDataPath.glob('*.cnt')

# # preprocessing parameters
# l_freq = 0.1
# h_freq = 75.0
# rsfreq = 200

# # channel number * rsfreq
# chunks = (62, rsfreq)

# dataset = h5Dataset(savePath, 'dataset')
# for cntFile in group:
#     print(f'processing {cntFile.name}')
#     eegData, chOrder = preprocessing_cnt(cntFile, l_freq, h_freq, rsfreq)
#     chOrder = [s.upper() for s in chOrder]
#     eegData = eegData[:, :-10*rsfreq]
#     grp = dataset.addGroup(grpName=cntFile.stem)
#     dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

#     # dataset attributes
#     dataset.addAttributes(dset, 'lFreq', l_freq)
#     dataset.addAttributes(dset, 'hFreq', h_freq)
#     dataset.addAttributes(dset, 'rsFreq', rsfreq)
#     dataset.addAttributes(dset, 'chOrder', chOrder)

# dataset.save()

import mne
from transforms.preprocess import (
    L_FREQ_FILTER, H_FREQ_FILTER, TARGET_RATE, WINDOW_SECONDS,
    process_sample as preprocess_sample
)
from concurrent.futures import ProcessPoolExecutor, as_completed


def ignore_last_seconds(data: np.ndarray, seconds: int, freq: int=200) -> np.ndarray:
    return data[:, :-seconds*freq]

def pack_sample(path: Path) -> tuple[mne.io.RawArray, Path]:
    try:
        return preprocess_sample(mne.io.read_raw_edf(path, preload=True)), path
    except Exception as e:
        print(f"Skipping due to error reading {path}: {e}")
        return None, path



def make_h5dataset_from_edfs_sync(edfs: list[Path], save_path: Path) -> h5Dataset:
    dataset = h5Dataset(save_path.parent, save_path.stem)
    for file in tqdm(edfs, desc="Processing EDF files"):
        print(f'Processing {file.stem}...')
        sample, _ = pack_sample(file)
        if sample is None:
            continue
        
        if file.stem in dataset.h5py_file.keys():
            continue
        ch_order, eeg_data = sample.ch_names, sample.get_data(units='uV')
        data = ignore_last_seconds(eeg_data, WINDOW_SECONDS, freq=TARGET_RATE)
        if data.shape[1] == 0:
            print(f'Skipping {file.stem} due to empty data')
            continue

        file_group = dataset.addGroup(grpName=file.stem)
        dset = dataset.addDataset(
            file_group, 'eeg',
            data,
            chunks=(len(ch_order), TARGET_RATE)
        )
        # Add pre-processing attributes
        dataset.addAttributes(dset, 'lFreq', L_FREQ_FILTER)
        dataset.addAttributes(dset, 'hFreq', H_FREQ_FILTER)
        dataset.addAttributes(dset, 'rsFreq', TARGET_RATE)
        dataset.addAttributes(dset, 'chOrder', list(map(str.upper, ch_order)))
        dataset.addAttributes(dset, 'origin_file', file.name)
        unreachable = gc.collect()
        print(f"Unreachable objects: {unreachable}")
    dataset.save()
    return dataset


def multiproc_make_h5dataset_from_edfs(edfs: list[Path], save_path: Path) -> h5Dataset:
    dataset = h5Dataset(save_path.parent, save_path.stem)
    process_sample_futures = []
    with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=24, mp_context=multiprocessing.get_context('spawn')) as executor:
        for file in edfs:
            process_sample_futures.append(
                executor.submit(pack_sample, file)
            )

        for future in tqdm(as_completed(process_sample_futures), total=len(process_sample_futures), desc="Processing EDF files"):
            if future.exception():
                print(f"Error processing {future.exception()}")
                executor.shutdown()
                raise future.exception()

            sample, file = future.result()
            if sample is None:
                continue

            sample: mne.io.RawArray
            file: Path

            if file.stem in dataset.h5py_file.keys():
                print(f'Duplicate file: {file.stem}, skipping...')
                continue

            file_group = dataset.addGroup(grpName=file.stem)
            ch_order, eeg_data = sample.ch_names, sample.get_data(units='uV')
            #### Add eeg data for each file
            dset = dataset.addDataset(
                file_group, 'eeg',
                ignore_last_seconds(eeg_data, WINDOW_SECONDS, freq=TARGET_RATE),
                chunks=(len(ch_order), TARGET_RATE)
            )
            #### Add pre-processing attributes
            dataset.addAttributes(dset, 'lFreq', L_FREQ_FILTER)
            dataset.addAttributes(dset, 'hFreq', H_FREQ_FILTER)
            dataset.addAttributes(dset, 'rsFreq', TARGET_RATE)
            dataset.addAttributes(dset, 'chOrder', list(map(str.upper, ch_order)))
            dataset.addAttributes(dset, 'origin_file', file.name)
            unreachable = gc.collect()
            print(f"Unreachable objects: {unreachable}")
    dataset.save()
    return dataset

if __name__ == '__main__':
    path = '/home/ubuntu/sami-workbench-az/tuh2500/raw/tuar/v3.0.1/edf/01_tcp_ar'
    dst_path = '/home/ubuntu/sami-workbench-az/tuh2500/hdf5/tuar_01_tcp_ar.h5'
    edfs = list(Path(path).glob('*.edf'))
    multiproc_make_h5dataset_from_edfs(edfs, Path(dst_path))