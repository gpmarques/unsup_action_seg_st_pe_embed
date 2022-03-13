""" features module

This module implements all features related classes and methods

...
Classes
-------
Features
    Class used to represent and manage the features extracted of a video by some
    feature extractor

"""
from npy import Npy
import numpy as np
import os
from sklearn.decomposition.pca import PCA

class Features:
    """ Class that represents the features extracted from a video

    Attributes
    ----------
    _path: str
        The path to the video of this features

    path: str
        The path where this features are stored

    _extractor: str
        The name of the extractor that extracted this features

    has_features: bool
        Indicates whether this features have already been extracted

    npy_io: Npy
        Object that manages the read and write functionalities of npy files,
        which is the file type that features are stored

    FEATURE_BASE_FNAME: str
        All npy files that store features have to have this suffix

    Methods
    ------

    __call__()
        Reads the path attribute with npy_io and returns a numpy array

    write(ith: int, features: np.ndarray)
        Writes features to path with the prefix XXXX with ith
    """

    # suffix of all features written by this module
    FEATURE_BASE_FNAME = "features.npy"

    def __init__(self, path: str, extractor: str):
        """
        Parameters
        ----------
        path: str
            path to the video for the features extracted

        str: str
            The name of the extractor that extracted this features

        npy_io: Npy
            Object responsible for all i/o operations with npy files
        """
        self._path = path
        self._extractor = extractor
        self.npy_io = Npy()

    @property
    def path(self):
        """ Path to this features """
        if os.path.isdir(self._path+"_features") is False:
            os.mkdir(self._path+"_features")
        return os.path.join(self._path+"_features", self._extractor)

    @property
    def has_features(self):
        """ Checks if this features exist """
        features_count = self.npy_io.count_npy_files(self.path)
        return features_count > 0

    def __call__(self, with_pe: bool = False, reduce_dim: bool = False) -> np.ndarray:
        """ Reads this features with the npy_io object in path """
        features = self.npy_io.read(self.path)
        
        if reduce_dim:
            features = PCA(n_components=.999999).fit_transform(features)
            
        if with_pe:
            return self._positional_encoding(features)
        
        return features

    def write(self, ith: int, features: np.ndarray):
        """ Writes features with ith prefix

        It first checks if the path where this features should be stored exist,
        if it doesn't, creates it. Then creates the file name of the feature
        passed, which is of the form XXXX_features.npy.

        Parameters
        ----------
        ith: int
            The prefix of a feature filename

        features: np.ndarray
            Numpy array storing a feature
        """
        if os.path.isdir(self.path) is False:
            os.mkdir(self.path)

        fname = f"{ith:04}_{self.FEATURE_BASE_FNAME}"
        self.npy_io.write(self.path, fname, features)

    def _positional_encoding(self, data: np.ndarray) -> np.ndarray:
        """
        Adds positional encoding in a numpy array. Positional encoding is a technique
        to inject order information into data, it was proposed in the paper Attention 
        Is All you Need, for more info: https://arxiv.org/abs/1706.03762

        Parameters
        ----------
        data: np.ndarray
            data that the positional encoding will be added

        Returns
        -------
        np.ndarray
            The input + the positional encoding
        """
        def get_sinusoid_encoding_table(length, d_model):
            '''  '''

            def cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

            def get_posi_angle_vec(position):
                return [cal_angle(position, hid_j) for hid_j in range(d_model)]

            sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(length)])

            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

            return sinusoid_table

        d_model = data.shape[1]
        length = data.shape[0]

        pe = get_sinusoid_encoding_table(length, d_model)

        return data + pe
