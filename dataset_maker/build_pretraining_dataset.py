
import os
import mne
from mne.filter import _triage_filter_params
import numpy as np
import pickle

import hashlib
import inspect
from pathlib import Path
from functools import wraps
from typing import Generator
from toolz import pipe, filter, curry, keyfilter, complement, map, take
from toolz.curried import pipe, filter, mapcat, groupby, keyfilter, map

from data_processor.dataset import ShockDataset
from transforms.preprocess import filter_channels, L_FREQ_FILTER, H_FREQ_FILTER
from dataset_maker.make_h5dataset_for_pretrain import multiproc_make_h5dataset_from_edfs, make_h5dataset_from_edfs_sync

HDF5_ROOT_PATH  = '/home/ubuntu/sami-workbench-az/tuh2500/hdf5'

def cache_from_disk(cache_dir: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            # consistent hashing across runs
            _hash = hashlib.md5(pickle.dumps(tuple(bound.arguments.items()))).hexdigest()
            print(f'hash: {_hash}')
            cache_path = Path(cache_dir) / Path(f'{_hash}.pkl')
            # cache hit
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                return pickle.loads(cache_path.read_bytes())
            # cache miss
            result = func(*args, **kwargs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pickle.dumps(result))
            return result

        return wrapper
    return decorator


# NOTE Implicitly assumes that each sequence (e.g. transformer patch) is 1 second long
@curry
def within_sequence_length(sequence_length: int, max_distance: int, sample: mne.io.RawArray) -> bool:
    sample_duration_seconds = sample.n_times / sample.info['sfreq']
    sample_sequence_length = sample_duration_seconds * len(sample.ch_names)
    # sample sequence length can be within max_distance of target sequence length
    return sample_sequence_length + max_distance >= sequence_length

@curry
def can_bandpass_safely(l_freq: float, h_freq: float, sample: mne.io.RawArray) -> bool:
    datalike = np.zeros((len(sample.ch_names), sample.n_times))
    datalike, *_, filter_length, _, _, _ =_triage_filter_params(
        datalike, sfreq=sample.info['sfreq'], l_freq=l_freq, h_freq=h_freq,
        #### NOTE Everything below this point is default and assume that
        #### our actual data processing will be default too.
        l_trans_bandwidth="auto", h_trans_bandwidth="auto", filter_length="auto",
        method="fir", phase="zero", fir_window="hamming", fir_design="firwin"
    )
    is_ok = filter_length <= datalike.shape[-1]
    if not is_ok:
        print(f"Filter length {filter_length}, with duration {filter_length / sample.info['sfreq']} seconds,"
              f"is greater than sample duration {sample.n_times / sample.info['sfreq']} seconds with length {sample.n_times} ",
              f"at {sample}")
    return is_ok

def edfs_from_dir(dir_path: Path) -> Generator[Path, None, None]:
    for p in dir_path.glob('**/*.edf'):
        print(f'yielding {p}')
        yield p
    # yield from dir_path.glob('**/*.edf')

def partial_load_sample(edf_path: Path) -> mne.io.RawArray:
    return mne.io.read_raw_edf(edf_path, preload=False)

def num_valid_channels(edf: mne.io.RawArray) -> int:
    try: return len(filter_channels(edf).ch_names)
    except Exception as e: print(f"Skipping due to error reading {edf}: {e}"); return 0

def hdf5_path_from_num_channels(num_channels: int) -> Path:
    return Path(HDF5_ROOT_PATH) / Path(f'{num_channels}channels.hdf5')

def channels_hdf5_exists(num_channels: int) -> bool:
    return hdf5_path_from_num_channels(num_channels).exists()

@curry
def take_percent(percent: float, edfs: list[mne.io.RawArray]) -> list[mne.io.RawArray]:
    return take(int(len(edfs) * percent), edfs)

@curry
@cache_from_disk("./caches/")
def parse_valid_edfs(sequence_length: int, l_freq: int, h_freq: int, clip_to_percent: float, dataset_dir: str) -> list[mne.io.RawArray]:
    # NOTE the caching implies that output needs to be deterministic. This means clip_to_percent cannot cause
    # the edf filepaths to be shuffled. Since Path.glob does not guarantee consistent ordering, we sort the edfs.
    # This exhausts the generator into a list, but should not affect performance.
    assert clip_to_percent >= 0. and clip_to_percent <= 1.
    return pipe(
        Path(dataset_dir),
        edfs_from_dir,
        sorted,
        take_percent(clip_to_percent),
        map(partial_load_sample),
        filter(can_bandpass_safely(l_freq, h_freq)),
        filter(within_sequence_length(sequence_length, 30)),
        list
    )


def build_pretraining_dataset(
    dataset_dirs: list[str],
    sequence_length: int,
    stride_size_seconds: int,
    start_percentage: float, end_percentage: float=1,
    l_freq: int=L_FREQ_FILTER, h_freq: int=H_FREQ_FILTER,
    sampling_rate: int=200,
    root_path: str = HDF5_ROOT_PATH,
    clip_to_percent: float = 1.
) -> tuple[list[ShockDataset], list[list[str]]]:
    global HDF5_ROOT_PATH
    HDF5_ROOT_PATH = root_path

    # Keep all valid edfs
    valid_edfs: dict[int, list[mne.io.RawArray]] = pipe(
        dataset_dirs,
        mapcat(parse_valid_edfs(sequence_length, l_freq, h_freq, clip_to_percent)),
        groupby(num_valid_channels),  
        keyfilter(bool), # keep channels > 0
        dict
    )

    exist, not_exist = (
        keyfilter(channels_hdf5_exists, valid_edfs),
        keyfilter(complement(channels_hdf5_exists), valid_edfs)
    )
    
    for num_channels, edfs in not_exist.items():
        # issue is that make_h5dataset needs paths and not objects
        make_h5dataset_from_edfs_sync(
            [
                Path(filename)
                for edf in edfs 
                for filename in edf.filenames
            ],
            hdf5_path_from_num_channels(num_channels)
        )
        exist[num_channels] = edfs

    return [
        ShockDataset(
            [hdf5_path_from_num_channels(num_channels)],
            time_window := sequence_length // num_channels,
            stride_size_hz := stride_size_seconds * sampling_rate,
            start_percentage, end_percentage
        )
        for num_channels in exist.keys()
    ]

# if __name__ == "__main__":
#     daraset_dir = '/home/ubuntu/sami-workbench-az/tuh2500/raw/tuar/v3.0.1/'
#     build_pretraining_dataset(
#         dataset_dirs=[daraset_dir],
#         sequence_length=256,
#         stride_size_seconds=1,
#         start_percentage=0.03,
#         end_percentage=0.97
#     )

if __name__ == '__main__':
    ### NOTE Pretrain val
    build_pretraining_dataset(
        dataset_dirs=[
            '/home/ubuntu/sami-workbench-az/tuh2500val/raw/tuab'
        ],
        sequence_length=256,
        stride_size_seconds=4,
        start_percentage=0.01,
        end_percentage=0.99,
        l_freq=0.5,
        root_path='/home/ubuntu/sami-workbench-az/tuh2500val/hdf5',
        clip_to_percent=0.2  # 20% of tuab 
    )
    ### NOTE Pretrain
    # build_pretraining_dataset(
    #     dataset_dirs=[
    #         '/home/ubuntu/sami-workbench-az/tuh2500/raw/tuar/v3.0.1/',
    #         '/home/ubuntu/sami-workbench-az/tuh2500/raw/tusl/v2.0.1/',
    #         '/home/ubuntu/sami-workbench-az/tuh2500/raw/tuep/v2.0.1/',
    #         '/home/ubuntu/sami-workbench-az/tuh2500/raw/tusz/edf/dev/',
    #         '/home/ubuntu/sami-workbench-az/tuh2500/raw/tusz/edf/train/',
    #     ],
    #     sequence_length=256,
    #     stride_size_seconds=4,
    #     start_percentage=0.01,
    #     end_percentage=0.99,
    #     l_freq=0.5
    # )