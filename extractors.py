""" Extractors module

This module implements all feature extraction related classes and methods

...
Classes
-------
ExtractionStrategy
    Abstract class used to represent a generic feature extraction strategy

SlowFastStrategy
    Concrete class used to represent the SlowFast feature extraction strategy

I3DStrategy
    Concrete class used to represent the I3D feature extraction strategy

ExtractorFactory
    Class used to represent a feature extractor factory, it contains all feature
    extraction strategies available and it makes them easily accessible

"""
from abc import ABC, abstractmethod
import numpy as np
from gluoncv.data.transforms import video as video_tranforms
from gluoncv.model_zoo import get_model
from mxnet import nd, gpu
import math
from enum import Enum

from video import Video
from dino import Dino
import torch


class ExtractionStrategy(ABC):
    """
    A class used to represent an abstract feature extraction strategy

    ...
    Attributes
    ----------
    model: object
        object that represents a model with feature extraction capabilities

    input_len: int
        Number of frames a model input must have

    FRAME_SIDE_SIZE: int
        The length of the height and width of an input frame a model expects

    IMAGENET_MEAN: list
        List with three elements, where each is the mean of each image channel
        in RGB convention from the ImageNet dataset

    IMAGENET_SD: list
        List with three elements, where each is the standard deviation of each
        image channel in RGB convention from the ImageNet dataset

    Methods
    -------
    extract(video: Video)
        Method that extract the features from a video object

    """

    model = NotImplemented  # model to extract features
    input_len = NotImplemented  # input clip frames length
    name = NotImplemented  # model name
    FRAME_SIDE_SIZE = 180
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_SD = [0.229, 0.224, 0.225]

    @abstractmethod
    def _sample_frames(self):
        """
        Subclasses of ExtractionStrategy must implement this method. This method
        is supposed to create a list of indexes, where each index corresponds to
        a frame from a video object.
        """
        pass

    def _preprocess(self, frames: np.ndarray, input_len: int) -> np.ndarray:
        """
        This method performs all preprocess operations and transformation that
        frames must pass through

        Parameters
        ----------
        frames: np.ndarray
            Numpy array of shape (N, L, H, W, C), where N is the number of clips,
            L is the number of frames each clip has, H and W are the height and
            width of each frame and C are image channels

        clip_len: int
            Number of frames a model input must have

        Returns
        -------
        clips_input: np.ndarray
            Preprocessed numpy array with all input frames
        """
        _, h, w, _ = frames.shape
        self.FRAME_SIDE_SIZE = min(h, w)
        transform_fn = video_tranforms.VideoGroupValTransform(
            size=self.FRAME_SIDE_SIZE,
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_SD)
        clips_input = transform_fn(frames)
        clips_input = np.stack(clips_input, axis=0)
        clips_input = clips_input.reshape((-1,) + (input_len, 3, self.FRAME_SIDE_SIZE, self.FRAME_SIDE_SIZE))
        clips_input = np.transpose(clips_input, (0, 2, 1, 3, 4))
        return nd.array(clips_input, gpu())

    def extract(self, video: Video) -> None:
        """
        This method extracts the features from a video object. First it retrieves
        all frames indexes according to the frame sampling method, then, for each
        clip's indexes it gets the frames from the video object, preprocess them,
        extract the features and saves them

        Parameters
        ----------
        video: Video
            Video object that is going to have its features extracted
        """
        frames_id_list = self._sample_frames(len(video))

        for i, ith_frames_id_list in enumerate(frames_id_list):
            frames = self._preprocess(video(ith_frames_id_list).asnumpy(),
                                      input_len=self.input_len)

            features = self.model(frames).asnumpy()

            video.features.write(i, features)


class SlowFastStrategy(ExtractionStrategy):
    """
    A class used to represent the feature extraction strategy that uses the Slowfast
    model. The Gluon library has many version of the SlowFast model, the one used
    here is the 8x8, resnet50, trained to classify the Kinetics400 dataset.

    ...
    Attributes
    ----------
    model: object
        object that represents a model with feature extraction capabilities

    input_len: int
        Number of frames a model input must have

    FRAME_SIDE_SIZE: int
        The length of the height and width of an input frame a model expects

    IMAGENET_MEAN: list
        List with three elements, where each is the mean of each image channel
        in RGB convention from the ImageNet dataset

    IMAGENET_SD: list
        List with three elements, where each is the standard deviation of each
        image channel in RGB convention from the ImageNet dataset

    MODEL_NAME: str
        Name, that the method get_model from Gluon's library, uses to access this
        model's pretrained weights and architecture

    Methods
    -------
    extract(video: Video)
        Method that extract the features from a video object

    """
    MODEL_NAME = 'slowfast_8x8_resnet50_kinetics400'

    def __init__(self, clip_len):
        self.model = get_model(self.MODEL_NAME, nclass=400,
                               pretrained=True, feat_ext=True, ctx=gpu())
        self.clip_len = clip_len
        self.input_len = self.clip_len + int(self.clip_len / 2) 
        self.name = ExtractorFactory.SLOWFAST

    def _sample_frames(self, video_len: int) -> list:
        """
        This method creates a list of list of frame indexes, where each nested
        list contains the frames indexes from a clip input. Each clip input list
        of indexes contains the indexes for the two pathways of the SlowFast model.

        Parameters
        ----------
        video_len: int
            The total number of frames the video this method is generating the
            frames indexes from

        Returns
        -------
        frames_id_list: list
            list of lists with the frames indexes of each input clip
        """
        n_batches = math.floor(video_len / self.clip_len)

        frames_id_list = [list(range((i-1)*self.clip_len, i*self.clip_len, 1)) +  # fast pathway frames
                          list(range((i-1)*self.clip_len, i*self.clip_len, 2))   # slow pathway frames
                          for i in range(1, n_batches + 1)]

        if video_len % self.clip_len > 0:  # make sure that all frames are used
            rest_frame_id_list = list(range(video_len - self.clip_len, video_len, 1)) +\
                                 list(range(video_len - self.clip_len, video_len, 2))
            frames_id_list.append(rest_frame_id_list)

        return frames_id_list


class I3DStrategy(ExtractionStrategy):
    """
    A class used to represent the feature extraction strategy that uses the I3D
    model. The Gluon library has many version of the I3D model, the one used
    here is the non-local 10, resnet50, trained to classify the Kinetics400
    dataset.

    ...
    Attributes
    ----------
    model: object
        object that represents a model with feature extraction capabilities

    clip_len: int
        Number of frames a model input must have

    FRAME_SIDE_SIZE: int
        The length of the height and width of an input frame a model expects

    IMAGENET_MEAN: list
        List with three elements, where each is the mean of each image channel
        in RGB convention from the ImageNet dataset

    IMAGENET_SD: list
        List with three elements, where each is the standard deviation of each
        image channel in RGB convention from the ImageNet dataset

    MODEL_NAME: str
        Name, that the method get_model from Gluon's library, uses to access this
        model's pretrained weights and architecture

    Methods
    -------
    extract(video: Video)
        Method that extract the features from a video object

    """
    MODEL_NAME = 'i3d_nl10_resnet50_v1_kinetics400'

    def __init__(self):
        self.model = get_model(self.MODEL_NAME, nclass=400,
                               pretrained=True, feat_ext=True, ctx=gpu())
        self.input_len = 32
        self.name = ExtractorFactory.I3D

    def _sample_frames(self, video_len: int) -> list:
        """
        This method creates a list of list of frame indexes, where each nested
        list contains the frames indexes from a clip input.

        Parameters
        ----------
        video_len: int
            The total number of frames the video this method is generating the
            frames indexes from

        Returns
        -------
        frames_id_list: list
            list of lists with the frames indexes of each input clip
        """
        n_batches = math.floor(video_len / self.input_len)

        frames_id_list = [list(range((i-1)*self.input_len, i*self.input_len, 1))
                          for i in range(1, n_batches + 1)]

        if video_len % self.input_len > 0:  # make sure that all frames are used
            rest_frame_id_list = list(range(video_len - self.input_len, video_len, 1))
            frames_id_list.append(rest_frame_id_list)

        return frames_id_list

class DinoStrategy(ExtractionStrategy):
    
    def __init__(self, BATCH_SIZE=50):
        self.model = Dino()
        self.BATCH_SIZE = BATCH_SIZE
    
    def _sample_frames(self, video_len: int) -> list:
        n_batches = math.floor(video_len / self.BATCH_SIZE)
        frames_id_list = [list(range((i-1)*self.BATCH_SIZE, i*self.BATCH_SIZE, 1))
                          for i in range(1, n_batches + 1)]

        if video_len % self.BATCH_SIZE > 0:  # make sure that all frames are used
            rest_frame_id_list = list(range(video_len - self.BATCH_SIZE, video_len, 1))
            frames_id_list.append(rest_frame_id_list)

        return frames_id_list
    
    def _preprocess(self, frames: torch.Tensor, input_len: int) -> np.ndarray:
        return frames.to(torch.float32).permute(0, 3, 2, 1)
    
    def extract(self, video: Video) -> None:
        frames_id_list = self._sample_frames(len(video))
        
        for i, ith_frames_id_list in enumerate(frames_id_list):
            frames = video(ith_frames_id_list, as_tensor=True)
            features = self.model(self._preprocess(frames, None))
            video.features.write(i, features.cpu().detach().numpy())

class ExtractorFactory(Enum):
    """
    Class used to represent a feature extraction strategy factory

    Attributes
    ----------
    Slowfast: enum member
        Enum member representing the SlowFastStrategy class

    I3D: enum member
        Enum member representing the I3DStrategy class

    Methods
    -------
    values_list()
        Returns a list with each enum member value

    get(cluster_type: str)
        Returns the ExtractionStrategy object matching the extraction_type para-
        -meter passed
    """
    SLOWFAST = "Slowfast"
    I3D = "I3D"
    DINO = "DINO"

    @staticmethod
    def values_list() -> list:
        """ Returns a list with each enum member value """
        return [ctype.value for ctype in ExtractorFactory]

    @staticmethod
    def get(extraction_type: str) -> ExtractionStrategy:
        """ Returns the ExtractionStrategy object matching the extraction_type
        parameter passed

        Parameters
        ----------
        extraction_type: str
            String with the value of some enum member from the ExtractorFactory

        Returns
        -------
        ExtractionStrategy
            Returns the ExtractionStrategy class of the extraction_type parameter
        """
        return {
            ExtractorFactory.SLOWFAST.value: SlowFastStrategy,
            ExtractorFactory.I3D.value: I3DStrategy,
            ExtractorFactory.DINO.value: DinoStrategy,
        }[extraction_type]
