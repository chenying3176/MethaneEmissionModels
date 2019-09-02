# Comparison of methane emission models
This repository contains code for the paper Comparison of methane emission models of reservoirs on a regional field scale: Performance and adaptation of models with different experimental datasets

## Dependencies

Make sure the following packages have already been installed before moving on:
* Python3
* Scikit-learn
* Numpy

## install
Clone the repo

```shell
## Download the repository
git clone https://github.com/chenying3176/CAU.git
cd CAU

```

## Usage
Your directory tree should look like this:
```
${POSE_ROOT}
├── exp_data
      ├── dataset_sample.xlsx
├── scripts 
      ├── training_and_test.py
      ├── data_processing.py
      └── data_processing_util.py
└── README.md
```

### data processing

run the following command:

```
python scripts/data_processing.py
```

after this command, your directory tree would look like this:
```
${POSE_ROOT}
├── exp_data
      ├── dataset_sample.xlsx
      ├── beijing
      └── beijing_others
├── scripts 
      ├── training_and_test.py
      ├── data_processing.py
      └── data_processing_util.py
└── README.md
```
the data that will be used to train and test have been saved under the tow new files: 'beijing' and 'beijing_others'

### train

run the following command:
```
python scripts/training_and_test.py
```

