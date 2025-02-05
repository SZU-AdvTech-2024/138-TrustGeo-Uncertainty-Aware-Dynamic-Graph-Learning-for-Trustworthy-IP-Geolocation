# TrustGeo: Uncertainty-Aware Dynamic Graph Learning for Trustworthy IP Geolocation

## Basic Usage

### 1. 环境设置

首先，创建并激活一个新的 Anaconda 虚拟环境，并安装所需的库：

代码需要 `python 3.8.13`, `PyTorch 1.12.1`,  `cudatoolkit 11.6.0`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```
#创建虚拟环境
conda create --name TrustGeo python=3.8.13

# 激活环境
conda activate TrustGeo

# 下载插件
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# 下载所需库
conda install numpy pandas
pip install scikit-learn
```

### 运行代码
### 下载数据集
https://github.com/ICDM-UESTC/TrustGeo/tree/main/datasets
#下载好之后需要将数据集解压到dataset文件夹
```
# Open the "TrustGeo" folder
cd TrustGeo
###加载数据集
# data preprocess (executing IP clustering). 
python preprocess.py --dataset "New_York"
python preprocess.py --dataset "Los_Angeles"
python preprocess.py --dataset "Shanghai"

# run the model TrustGeo
python main.py --dataset "New_York" --lr 5e-3 --dim_in 30 --lambda1 7e-3
python main.py --dataset "Los_Angeles" --lr 3e-3 --dim_in 30 --lambda1 7e-3
python main.py --dataset "Shanghai" --lr 0.0015 --dim_in 51 --lambda1 1e-3

# load the checkpoint and then test
python test.py --dataset "New_York" --lr 5e-3 --dim_in 30 --lambda1 7e-3 --load_epoch 400
python test.py --dataset "Los_Angeles" --lr 3e-3 --dim_in 30 --lambda1 7e-3 --load_epoch 400
python test.py --dataset "Shanghai" --lr 0.0015 --dim_in 51 --lambda1 1e-3 --load_epoch 200
```

## The description of hyperparameters used in main.py

| Hyperparameter   | Description                                                  |
| :--------------- | ------------------------------------------------------------ |
| model_name       | the name of model                                            |
| dataset          | the dataset used by main.py                                  |
| lambda           | the trade-off coefficient of loss function                   |
| lr               | learning rate                                                |
| harved_epoch     | when how many consecutive epochs the performance does not increase, the learning rate is halved |
| early_stop_epoch | when how many consecutive epochs the performance does not increase, the training stops. |
| saved_epoch      | how many epochs to save checkpoint for the testing           |
| seed             | the random number seed used for parameter initialization during training |
| dim_in           | the dimension of input data                                  |



## Folder Structure

```tex
└── TrustGeo
	├── datasets # Contains three large-scale real-world street-level IP geolocation datasets.
	│	|── New_York # Street-level IP geolocation dataset collected from New York City including 91,808 IP addresses.
	│	|── Los_Angeles # Street-level IP geolocation dataset collected from Los Angeles including 92,804 IP addresses.
	│	|── Shanghai # Street-level IP geolocation dataset collected from Shanghai including 126,258 IP addresses.
	├── lib # Contains model implementation files
	│	|── layers.py # The code of the attention mechanism.
	│	|── model.py # The core source code of the proposed TrustGeo
	│	|── sublayers.py # The support file for layer.py
	│	|── utils.py # Auxiliary functions, including the code of view fusion
	├── asset # Contains saved checkpoints and logs when running the model
	│	|── log # Contains logs when running the model 
	│	|── model # Contains the saved checkpoints
	├── preprocess.py # Preprocess dataset and execute IP clustering for the model running
	├── main.py # Run model for training and test
	├── test.py # Load checkpoint and then test
	└── README.md
```

## Dataset Information

The "datasets" folder contains three subfolders corresponding to three large-scale real-world street-level IP geolocation    datasets collected from New York City, Los Angeles, and Shanghai. There are three files in each subfolder:

- data.csv    *# features (including attribute knowledge and network measurements) and labels (longitude and latitude) for street-level IP geolocation* 
- ip.csv    *# IP addresses*
- last_traceroute.csv    *# last four routers and corresponding delays for efficient IP host clustering*

The detailed **columns and description** of data.csv in the New York dataset are as follows:

#### New York  

| Column Name                     | Data Description                                             |
| ------------------------------- | ------------------------------------------------------------ |
| ip                              | The IPv4 address                                             |
| as_mult_info                    | The ID of the autonomous system where IP locates             |
| country                         | The country where the IP locates                             |
| prov_cn_name                    | The state/province where the IP locates                      |
| city                            | The city where the IP locates                                |
| isp                             | The Internet Service Provider of the IP                      |
| vp900/901/..._ping_delay_time   | The ping delay from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._trace             | The traceroute list from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._tr_steps          | #steps of the traceroute from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._last_router_delay | The delay from the last router to the IP host in the traceroute list from probing hosts "vp900/901/..." |
| vp900/901/..._total_delay       | The total delay from probing hosts "vp900/901/..." to the IP host |
| longitude                       | The longitude of the IP (as the label)                           |
| latitude                        | The latitude of the IP host (as the label)                       |

PS: The detailed columns and description of data.csv in the other two datasets are similar to the New York dataset.

