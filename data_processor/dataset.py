import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset


list_path = List[Path]

class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm.
    
    This dataset reads EEG data from an HDF5 file containing multiple subjects' recordings.
    It splits the continuous EEG signals into overlapping windows/segments based on:
    - window_size: Length of each EEG segment (e.g. 200 timepoints)
    - stride_size: How many timepoints to slide the window (e.g. 1 for maximum overlap)

    The HDF5 structure is:
    file
    ├── subject1
    │   └── eeg [channels x timepoints]
    ├── subject2
    │   └── eeg [channels x timepoints] 
    └── ...
    """
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.file_path = file_path
        self.window_size = window_size
        self.stride_size = stride_size
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage

        self.hdf5_file = None
        self.length = None
        self._feature_size = None

        # Lists to track:
        self.file_subject_ids = []  # Subject IDs in the file
        self.file_subject_idx = []  # Starting index for each subject in the flattened dataset
        self.subject_local_idx = []  # Starting timepoint within each subject's data
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.hdf5_file = h5py.File(str(self.file_path), 'r')
        self.file_subject_ids = [i for i in self.hdf5_file]

        global_idx = 0
        for subject in self.file_subject_ids:
            self.file_subject_idx.append(global_idx) # the start index of the subject's sample in the dataset
            subject_len = self.hdf5_file[subject]['eeg'].shape[1]
            # Calculate how many windows we can extract from this subject's data
            total_sample_num = (subject_len-self.window_size) // self.stride_size + 1
            # Only use windows within the specified percentage range
            start_idx = int(total_sample_num * self.start_percentage) * self.stride_size 
            end_idx = int(total_sample_num * self.end_percentage - 1) * self.stride_size

            self.subject_local_idx.append(start_idx)
            global_idx += (end_idx - start_idx) // self.stride_size + 1
        self.length = global_idx

        self._feature_size = [i for i in self.hdf5_file[self.file_subject_ids[0]]['eeg'].shape]
        self._feature_size[1] = self.window_size

    @property
    def feature_size(self):
        return self._feature_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        """Get a single window of EEG data.
        
        Given a flat index idx, this:
        1. Finds which subject this window belongs to using binary search
        2. Calculates the starting timepoint within that subject's data
        3. Returns a window of size window_size starting at that timepoint
        
        For example, if idx=100, stride=1, window=200:
        - May map to subject 2's data starting at timepoint 50
        - Returns channels x 200 window of EEG data
        """
        subject_idx = bisect.bisect(self.file_subject_idx, idx) - 1
        item_start_idx = (idx - self.file_subject_idx[subject_idx]) * self.stride_size + self.subject_local_idx[subject_idx]
        return self.hdf5_file[self.file_subject_ids[subject_idx]]['eeg'][:, item_start_idx:item_start_idx+self.window_size]
    
    def free(self) -> None: 
        if self.hdf5_file:
            self.hdf5_file.close()
            self.hdf5_file = None
    
    def get_ch_names(self):
        return self.hdf5_file[self.file_subject_ids[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """Integrates multiple HDF5 files into a single dataset.
    
    This is a wrapper that combines multiple SingleShockDatasets into one large dataset.
    It maintains indices to map from a flat index to the correct file and position.
    """
    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0]._feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()
