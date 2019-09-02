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

Run the following command:

```
python scripts/data_processing.py
```

After this command, your directory tree would look like this:
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
The data that will be used to train and test have been saved under the tow new files: 'beijing' and 'beijing_others'

### train&test
Before strat training, you should add an empty folder called 'result' under the root, and your directory tree should look like this:
```
├── exp_data
      ├── dataset_sample.xlsx
      ├── beijing
      └── beijing_others
├── scripts 
      ├── training_and_test.py
      ├── data_processing.py
      └── data_processing_util.py
├── result
└── README.md
```
Then, to train and acquire the test result, run the following command:
```
python scripts/training_and_test.py
```
To get all the results in different region type, you can change a variation in main function of training_and_test.py:
```
region_type = 'DW'   # DW LIT
```
Similarly, you can change another variation to get all the results of different emission substance:
```
tag_type= 'C02'  # CO2  CH4 N2O 
```
Otherwise, the variation 'bank_name' indicate the region field scale:
```
bank_name='MY'   # MY,  all
```



