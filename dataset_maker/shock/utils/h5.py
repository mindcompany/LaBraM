import h5py
import numpy as np
from pathlib import Path

class h5Dataset:
    def __init__(self, path:Path, name:str) -> None:
        self.__name = name
        self.h5py_file = h5py.File(path / f'{name}.hdf5', 'a')
    
    def addGroup(self, grpName:str):
        try:
            return self.h5py_file.create_group(grpName)
        except Exception as e:
            print(f"Error creating group {grpName}: {e}")
            raise
    
    def addDataset(self, grp:h5py.Group, dsName:str, arr:np.array, chunks:tuple):
        return grp.create_dataset(dsName, data=arr, chunks=chunks)
    
    def addAttributes(self, src:'h5py.Dataset|h5py.Group', attrName:str, attrValue):
        src.attrs[f'{attrName}'] = attrValue
    
    def save(self):
        self.h5py_file.close()
    
    @property
    def name(self):
        return self.__name

