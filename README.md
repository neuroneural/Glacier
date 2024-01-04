# Glacier: Glass-Box Transformer for Interpretable Dynamic Neuroimaging


accepted at ICASSP 2023
Paper: https://ieeexplore.ieee.org/document/10097126


Usman Mahmood, Zengin Fu, Vince D. Calhoun, Sergey M. Plis

Gives SOTA classification performance on FBIRN, ABIDE, OASIS and ABCD datasets for schizophrenia, Autism, Dementia and Gender classification tasks. Outputs True or False for each subject for each dataset. 

The datasets required are ICA components. Please refer to the paper for data pre_processing details

Cannot upload data because of restrictions. Input data should be in the directore 'DataandLabels'. Input data should be of shape (n_subjects, n_components, n_time_points) e.g. (311, 100, 160) for FBIRN dataset. Target Labels should also be in the same directory

Tested on python 3.7, pytorch 1.7.1 cuda 11

#### Dependencies:
* PyTorch
* Scikit-Learn


### Installation 

```bash
# install PyTorch
git clone https://github.com/UsmanMahmood27/BrainGNN.git
cd Glacier
pip install -e .
pip install -r requirements.txt
```


