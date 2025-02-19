import os
import mne
import glob
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


mne.set_log_level(verbose=False)

BASE_PATH = '/home/ubuntu/sami-workbench-az/tuab/tuh_eeg_abnormal/v3.0.1/edf/train/abnormal/01_tcp_ar'


# NOTE These are from LaBram
DROP_CHANNELS = [
    'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'PULSE RATE', 'RESP ABDOMEN-REF',
    'ECG EKG-REF',
    'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF',
    'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF',
    'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF',
    'EEG EKG-REF', 'EEG PG2-REF', 'EEG PG1-REF',
    *(f'EEG {i}-REF' for i in range(20, 129))
]
CHANNEL_ORDER = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
]
##### Filters
L_FREQ_FILTER = 0.1
H_FREQ_FILTER = 75.0
NOTCH_FREQ = 50
TARGET_RATE = 200
RESAMPLE_N_JOBS = 4
UNITS = 'uV'
##### Data sampling
WINDOW_OVERLAP_SECONDS = 0
WINDOW_SECONDS = 10
##### Train/test/val split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.


def sliding_window(channel_data: np.ndarray, hz: int, num_seconds: int, overlap: int = 0) -> list[np.ndarray]:
    step = (window_size := hz * num_seconds) - (hz * overlap)
    n_windows = (channel_data.shape[1] - window_size) // step + 1
    return [
        channel_data[:,(i*step):(i*step)+window_size]
        for i in range(n_windows)
    ]


def filter_channels(sample: mne.io.RawArray) -> mne.io.RawArray:
    return sample\
        .drop_channels(set(DROP_CHANNELS) & set(sample.ch_names))\
        .reorder_channels([ch for ch in CHANNEL_ORDER if ch in sample.ch_names])


def process_sample(sample: mne.io.RawArray) -> mne.io.RawArray:
    sample.rename_channels(str.upper)    
    sample = filter_channels(sample)
    return sample\
        .filter(l_freq=L_FREQ_FILTER, h_freq=H_FREQ_FILTER)\
        .notch_filter(NOTCH_FREQ)\
        .resample(TARGET_RATE, n_jobs=RESAMPLE_N_JOBS)    

def load_and_process_file(path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # info = mne.io.read_raw_edf(path, preload=False)
    sample: mne.io.RawArray = mne.io.read_raw_edf(path, preload=True)
    processed_sample = process_sample(sample)

    descriptions_aligned: np.ndarray = np.array([''] * processed_sample.n_times, dtype=str)
    timestamps: np.ndarray = (processed_sample.annotations.onset * processed_sample.info['sfreq']).astype(np.uint64)
    
    for desc, timestamp in zip(
        descriptions := processed_sample.annotations.description,
        timestamps
    ):
        descriptions_aligned[int(max(timestamp-1, 0))] = desc

    # print(f'Descriptions for {path}: {set(descriptions)}')
    data: np.ndarray = processed_sample.get_data(units=UNITS)
    
    return (
        sliding_window(data, TARGET_RATE, WINDOW_SECONDS, WINDOW_OVERLAP_SECONDS),
        sliding_window(descriptions_aligned[np.newaxis, :], TARGET_RATE, WINDOW_SECONDS, WINDOW_OVERLAP_SECONDS)
    )

def plot_eeg(signal, title="EEG Signal", channels_to_plot=[0], figsize=(12, 4)):
    import matplotlib.pyplot as plt
    """
    signal: EEG array of shape (channels, time)
    channels_to_plot: list of indices of channels to plot
    """
    plt.figure(figsize=figsize)
    for ch in channels_to_plot:
        plt.plot(signal[ch], label=f"Channel {ch}")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.close()


def split_and_save(src_edf_filepath: str, dst_pkl_dir: str):
    windows, labels = load_and_process_file(src_edf_filepath)
    for i, (window, label) in enumerate(zip(windows, labels)):
        base_filename = os.path.basename(src_edf_filepath).replace('.edf', '')
        dst_filename = f"{base_filename}_{i}_label-0.pkl"
        with open(os.path.join(dst_pkl_dir, dst_filename), "wb") as f:
            # NOTE We don't have seizure start/end info so everything is 0 for now
            pickle.dump({"X": window, "y": 0}, f)
        print(f"Saved {dst_filename}")


def main(args):
    edf_files = glob.glob(os.path.join(args.stratus_path, "*.edf"))
    np.random.shuffle(edf_files)
    
    # NOTE: Split them into train/test/val sets.
    train_filepaths = edf_files[: int(len(edf_files) * TRAIN_RATIO)]
    val_filepaths = edf_files[
        int(len(edf_files) * TRAIN_RATIO) : int(len(edf_files) * (TRAIN_RATIO + VAL_RATIO))
    ]
    test_filepaths = edf_files[int(len(edf_files) * (TRAIN_RATIO + VAL_RATIO)) :]

    futures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for split, filepaths in [
            ("train", train_filepaths),
            ("val", val_filepaths),
            ("test", test_filepaths),
        ]:
            save_dir_path = os.path.join(args.stratus_path, split)
            if not os.path.exists(save_dir_path):
                print(f"Creating {save_dir_path}")
                os.makedirs(save_dir_path, exist_ok=True)

            for filepath in filepaths:
                futures.append(executor.submit(split_and_save, filepath, save_dir_path))

    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing .pkl for dataset..."):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process EDF files into PKL format")
    parser.add_argument(
        "--stratus_path",
        type=str,
        default="/home/ubuntu/sami-workbench-az/stratus",
        help="Path to the directory containing EDF files",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=15,
        help="Maximum number of workers for processing EDF files",
    )
    args = parser.parse_args()
    main(args)
