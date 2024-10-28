# Single-Image Depth Estimation Based on Fourier Domain Analysis

# IPR Project

This implementation is done on a Linux environment.


### Clone the Repository

Clone this repository and navigate into it:

```bash
git clone https://github.com/shiv57107/IPR_Project.git
cd IPR_Project
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Create Directories

``` bash
mkdir -p data/nyu_v2/train/depths
mkdir -p data/nyu_v2/train/images
mkdir -p data/nyu_v2/val/depths
mkdir -p data/nyu_v2/val/images
```
### Create Model Directories 
```bash
mkdir -p models/den_gen2_v122orig
mkdir -p models/pretrained_resnet
mkdir -p models/temp_v3
```
### Download the Small Dataset
```bash 
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
mv nyu_depth_v2_labeled.mat ./data/nyu_v2/
```

### Convert data
To Convert data into the required format 
``` bash
python converter.py
```
### Download the PreTrained ResNet
```
python helper.py
```
### Training the Model
To train the model, execute:
```bash
python run.py
```
#### Repeated Training
To repeatedly train the model, either change the experiment name in run.py or delete the trained model directories in the models folder, which will be named after the experiment.

Trained weights can be accessed here: [Gdrive](https://drive.google.com/file/d/1n39DKMLWwxZ4HNFSaF1HeAH4aP6Mz_7W/view?usp=sharing)

## Citation


```bibtex
@inproceedings{
  author = {Jae-Han Lee and Minhyeok Heo and Kyung-Rae Kim and Shih-En Wei and Chang-Su Kim},
  booktitle = {CVPR},
  title = {Single-Image Depth Estimation Based on Fourier Domain Analysis},
  year = {2018}
}
