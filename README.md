# Hand3D

![Teaser](teaser.png)

ColorHandPose3D is a Convolutional Neural Network estimating 3D Hand Pose from a single RGB Image. See the [original project page](https://lmb.informatik.uni-freiburg.de/projects/hand3d/) for the dataset used and additional information.

### Note!!!
I, Aman, have made a lot of modifications to this original code to make it closer to a production-ready app. There is still more to be done but the deciding factor here is speed. I've removed computation-hungry and unnecessary steps and also converted it into a live video pipeline. And I've optimized this whole codebase as much as I could, using multi-threading to increase camera FPS. Even after all these steps, on my CPU the inference takes 5 dumb seconds to process every single image.

## Recommended system
Recommended system (tested):
- Ubuntu 16.04.2 (xenial)
- Tensorflow 1.3.0 GPU build with CUDA 8.0.44 and CUDNN 5.1
- Python 3.5.2

Python packages used by the example provided and their recommended version:
- tensorflow==1.3.0
- numpy==1.13.0
- scipy==0.18.1
- matplotlib==1.5.3

## Usage: for web API
The function you need will be in the `api.py` file. Make sure this file is run only from within the repository (it's not a standalone program you can copy anywhere else) because it has many dependencies.

How to use this file?

1. Try to use a PNG image, and save it in a folder.
2. Call the `detect(img_path)` function with, you guessed it, the image path as an argument. It will do inference on the image and save the result as `results/detected_image.png`.
3. Then you can use this result image whichever way you want!


## Usage: For live webcam

- Download [data](https://lmb.informatik.uni-freiburg.de/projects/hand3d/ColorHandPose3D_data_v3.zip) and unzip it. Go into this folder and take out two of them, "results" and "weights", out into the main repository.
- Run the file `run_parallel.py` (the one with multi-threaded webcam pipeline). Wait for the camera feed to open up and then play with your hand gestures. It will not recognize the gesture as such, but it will be able to detect finger joints etc and draw them in a cool way.
- To end the stream, click on the video feed and hit 'Q' on your keyboard repeatedly until it stops. Not very elegant but that's how it is.

------------ Everything else is from the original documentation, for training etc purposes -------------------

## Preprocessing for training and evaluation
In order to use the training and evaluation scripts you need download and preprocess the datasets.

### Rendered Hand Pose Dataset (RHD)

- Download the dataset accompanying this publication [RHD dataset v. 1.1](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
- Set the variable 'path_to_db' to where the dataset is located on your machine
- Optionally modify 'set' variable to training or evaluation
- Run

		python3.5 create_binary_db.py
- This will create a binary file in *./data/bin* according to how 'set' was configured

### Stereo Tracking Benchmark Dataset (STB)
- For eval3d_full.py it is necessary to get the dataset presented in Zhang et al., ‘3d Hand Pose Tracking and Estimation Using Stereo Matching’, 2016
- After unzipping the dataset run

		cd ./data/stb/
		matlab -nodesktop -nosplash -r "create_db"
- This will create the binary file *./data/stb/stb_evaluation.bin*


## Network training
We provide scripts to train HandSegNet and PoseNet on the [Rendered Hand Pose Dataset (RHD)](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html).
In case you want to retrain the networks on new data you can adapt the code provided to your needs.

The following steps guide you through training HandSegNet and PoseNet on the Rendered Hand Pose Dataset (RHD).

- Make sure you followed the steps in the section 'Preprocessing'
- Start training of HandSegNet with training_handsegnet.py
- Start training of PoseNet with training_posenet.py
- Set USE_RETRAINED = True on line 32 in eval2d_gt_cropped.py
- Run eval2d_gt_cropped.py to evaluate the retrained PoseNet on RHD-e
- Set USE_RETRAINED = True on line 31 in eval2d.py
- Run eval2d.py to evaluate the retrained HandSegNet + PoseNet on RHD-e

You should be able to obtain results that roughly match the following numbers we obtain with Tensorflow v1.3:

eval2d_gt_cropped.py yields:

    Evaluation results:
    Average mean EPE: 7.630 pixels
    Average median EPE: 3.939 pixels
    Area under curve: 0.771


eval2d.py yields:

    Evaluation results:
    Average mean EPE: 15.469 pixels
    Average median EPE: 4.374 pixels
    Area under curve: 0.715

Because training itself isn't a deterministic process results will differ between runs.
Note that these results are not listed in the paper.



## Evaluation

There are four scripts that evaluate different parts of the architecture:

1. eval2d_gt_cropped.py: Evaluates PoseNet  on 2D keypoint localization using ground truth annoation to create hand cropped images (section 6.1, Table 1 of the paper)
2.  eval2d.py: Evaluates HandSegNet and PoseNet on 2D keypoint localization (section 6.1, Table 1 of the paper)
3.  eval3d.py: Evaluates different approaches on lifting 2D predictions into 3D (section 6.2.1, Table 2 of the paper)
3.  eval3d_full.py: Evaluates our full pipeline on 3D keypoint localization from RGB (section 6.2.1, Table 2 of the paper)

This provides the possibility to reproduce results from the paper that are based on the RHD dataset.
