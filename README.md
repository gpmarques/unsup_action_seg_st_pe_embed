# unsup_action_seg_st_pe_embed

Python implementation of the paper "Unsupervised Method for Video Action Segmentation Through Spatio-Temporal and Positional-Encoded Embeddings".

## Data

Download the Breakfast [dataset](https://drive.google.com/open?id=1jgSoof1AatiDRpGY091qd4TEKF-BUt6I) and the ground truth segmentations [files](https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E)

Download the INRIA Instructionals [dataset](https://www.di.ens.fr/willow/research/instructionvideos/data_new.tar.gz) videos and ground truth files


## Install dependencies

Install [conda](https://docs.conda.io/en/latest/miniconda.html)

Install [video_features](https://github.com/v-iashin/video_features) 

Install [decord](https://github.com/dmlc/decord.git) for efficient video reading

##### Using Conda, run:
```
conda env create -f environment.yml
conda activate as_env
jupyter-lab
``` 
## Run Experiment

First, execute the extract_features.ipynb file to extract video features using I3D and Slowfast; 

Run evaluation\_inria.ipynb file to execute the experiment on the INRIA dataset;

Run evaluation\_breakfast.ipynb file to execute the experiment on the Breakfast dataset.



