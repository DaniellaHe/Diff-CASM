# Diff-CASM: Vertebral Disease Classification with Context-Aware Abnormal Saliency Maps via Denoising Diffusion

## A Quick Overview 

![figure2](images/figure2_miccai_overview.pdf)

## Training & Evaluation

### Step 1. Requirements

In the terminal, execute the following command to create a Conda environment named `diffcasm`:

```
conda create --name diffcasm

# Activate the newly created environment:
conda activate diffcasm
```

You can use a file named `requirements.txt` to configure your `diffcasm` environment. This file contains all the necessary packages and their version information for the environment.

```
conda install --file requirements.txt
```

### Step 2. Dataset Generation

To generate the dataset required for your project, follow these steps:

```
# 1. Convert JSON to PNG format:
python ./data/VerTumor600/MRI_vertebrae/json_to_png.py

# 2. Split the dataset into training and testing sets:
python ./data/VerTumor600/MRI_vertebrae/split_train_test.py
```

### Step 3. Train the Diff-CASM

To train a model, run:

```
python diff_training_seg_training.py ARG_NUM=1
```

Here, `ARG_NUM` corresponds to the number related to the JSON argument file. These arguments are stored in `./configs/` and are named `args1.json`, for example.

Ensure that you have the necessary arguments properly configured in the corresponding JSON file located in the `./configs/` directory before running the training script. Adjust the arguments as needed to customize the training process according to your specific requirements.

### Step 4. Generate the saliency maps by Diff-CASM

```
python ./detection/diff_seg.py ARG_NUM=2
```

### Step 5. Train the classifier and evaluation

```
# 1. Crop the images:
python data/VerTumor600/cropped_vertebrae/crop_images.py

# 2. Train the classifier:
python train_two_branches.py
```

## License

This project is licensed under the terms of the MIT License.

## Acknowledgement

This work is built upon the [AnoDDPM](https://github.com/Julian-Wyatt/AnoDDPM), ACAT, DRAEM

Thanks to the following excellent works: [DiffMIC](https://github.com/scott-yjyang/DiffMIC), DDPM, Beat-GAN, [guided-diffusion](https://github.com/openai/guided-diffusion)

