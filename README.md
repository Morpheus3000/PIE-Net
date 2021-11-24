# Supporting code for PIENet

## Repository Structure
./test_outputs/ - Contains sample images to run the test on.
./Eval.py       - Evaluation code to run the model on a given folder containing the test images.
./Network.py    - Network definition of PIENet.
./Utils.py      - Supporting script providing convenient functions for loading and saving models, writing output images, etc.

## Requirements
Please install the following:
1. Pytorch (tested with version 1.0.1.post2) - Deep learning framework.
2. Tqdm                                      - Progress bar library.
3. Numpy                                     - Linear algebra library.
4. imageio                                   - Image loading library.
5. OpenCV                                    - Image Processing library.

Please download the model file (2.5GB) from: https://gofile.io/d/tk5c4B

## Inference
In the file Eval.py you can point to your custom image directory in L38. In
L39, the format of the image files to be looked for can be specified. The
output directory for the model can be set in L33. The script will create a new
folder if the folder doesn't exists.
Finally, in L36, set the location to the downloaded model file and save your changed.
The script can then be run from the command line as follows:
```
python Eval.py
```

The outputs are as follows:
*_img.png: Input image file.
*_pred_alb.png: The predicted albedo.
*_pred_shd.png: The predicted shading.
