# Learn to Adapt 

This repository is the re-implementation of ["Learn to Adapt for Generalized Zero-Shot Text Classification"](https://aclanthology.org/2022.acl-long.39.pdf) (ACL 2022 main conference). The reproducibility report can be found here under the name "report final lta".


## Requirements
1. First git clone  our repository.

   ```shell
   git clone https://github.com/Shrushti1999/Learn-To-Adapt.git
   ```

2. To setup the environment, we recommend to use a terminal that supports GPU.

   ```shell
   conda create -n lta python=3.7.9
   ```
   ```shell
   source ~/.bashrc
   ```
   Activate the conda environment `lta`.

   ```shell
   conda activate lta
   ```

3. After we setup basic conda environment, install `torch` and `gensim`.
   ```shell
   pip install torch
   ```
   ```shell
   pip install gensim==3.8.1
   ```
   Later install requirements.

   ```shell
   pip install -r requirements.txt
   ```

## Datasets and Files

1. Download `word2vec` English resource https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz, 
and put the unzip `bin` file to the directory `data/resources`.

2. Preprocess the dataset
   
   ```shell
   python data_preprocess.py
   ```

   And `pickle` files will be stored in `data/ver1/CLINC`. 

## Training and Evaluation

We use configuration files to store hyper-parameters for experiments in `config_clinc_LTSM.json`

To train the **Metric Learning** model, run this command:

```shell
python train.py -d 0 -st1 1 -c config_Clinc_LSTM.json
```

where `-st1` means step 1. If you want to run LTA w/o init, you do not need to run step 1 first. 


To train the LTA in the paper, run step 2:

```shell
python train.py -d 0 -st2 1 -c config_Clinc_LSTM.json
```

Or you can change any configurations in the `json` file, to test different hyper-parameters.

## Robustness

Cut the ```train_seen.csv``` and paste it in the Learn-To-Adapt folder. Run the ```Robustness.ipynb``` file to generate the noisy dataset. You can also see the dataset ```train_seen Robustness.cs ```

## Multilinguality

Cut the ```train_seen.csv``` and paste it in the Learn-To-Adapt folder. Run the ```Multilinguality.ipynb``` file to generate the multilingual dataset. You can also see the dataset ```train_seen Multilinguality.cs ```
 
# Reference
```
@inproceedings{zhang-etal-2022-learn,
    title = "Learn to Adapt for Generalized Zero-Shot Text Classification",
    author = "Zhang, Yiwen  and
      Yuan, Caixia  and
      Wang, Xiaojie  and
      Bai, Ziwei  and
      Liu, Yongbin",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.39",
    doi = "10.18653/v1/2022.acl-long.39",
    pages = "517--527",
    abstract = "Generalized zero-shot text classification aims to classify textual instances from both previously seen classes and incrementally emerging unseen classes. Most existing methods generalize poorly since the learned parameters are only optimal for seen classes rather than for both classes, and the parameters keep stationary in predicting procedures. To address these challenges, we propose a novel Learn to Adapt (LTA) network using a variant meta-learning framework. Specifically, LTA trains an adaptive classifier by using both seen and virtual unseen classes to simulate a generalized zero-shot learning (GZSL) scenario in accordance with the test time, and simultaneously learns to calibrate the class prototypes and sample representations to make the learned parameters adaptive to incoming unseen classes. We claim that the proposed model is capable of representing all prototypes and samples from both classes to a more consistent distribution in a global space. Extensive experiments on five text classification datasets show that our model outperforms several competitive previous approaches by large margins. The code and the whole datasets are available at https://github.com/Quareia/LTA.",
}
```

