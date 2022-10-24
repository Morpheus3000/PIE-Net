# PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition
This is the official model and network release for the paper:

P. Das, S. Karaoglu and T. Gevers, [PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition](https://ivi.fnwi.uva.nl/cv/pienet/assets/PIE_NET_CVPR_2022_main_paper.pdf), IEEE Conference of Computer Vision and Pattern Recognition (CVPR), 2022. The official project page can be found [here](https://ivi.fnwi.uva.nl/cv/pienet/). The pretrained model for the realworld evaluations can be downloaded from [here](https://uvaauas.figshare.com/articles/conference_contribution/real_world_model_t7/19940000)

Our model exploits illumination invariant features in an edge-driven hybrid CNN approach. The model is able to predict physically consistent reflectance and shading from a single input image, without the need for any specific priors. The network is trained without any specialised dataset or losses. 

![Propose network](/images/net_overview_github.png "The proposed network.")

The predicted reflectance by the network is able to minimise both soft and hard illumination effects. Conversely, the predicted shading minimises texture leakages.

![Our model's prediction](/images/Output_teaser_github.png "The proposed method.")

The network code, the pretrained model (trained only on synthetic outdoor data) and the fine tuned models on the MIT and IIW datasets are provided. Please download the pretrained and finetuned models from the project page.

## Repository Structure

The repository is provided in the same structure that the scripts expects the other supporting files.

./test_outputs/ - Contains sample images to run the test on.\
./Eval.py       - Evaluation code to run the model on a given folder containing the test images.\
./Network.py    - Network definition of PIENet.\
./Utils.py      - Supporting script providing convenient functions for loading and saving models, writing output images, etc.

## Requirements
Please install the following:
1. Pytorch (tested with version 1.0.1.post2) - Deep learning framework.
2. Tqdm                                      - Progress bar library.
3. Numpy                                     - Linear algebra library.
4. imageio                                   - Image loading library.
5. OpenCV                                    - Image Processing library.

## Inference
In the file Eval.py you can point to your custom image directory in L38. In
L39, the format of the image files to be looked for can be specified. The
output directory for the model can be set in L33. The script will create a new
folder if the folder doesn't exists.

Finally, in L36, set the location to the downloaded model file and save your changes.
The script can then be run from the command line as follows:
```
python Eval.py
```

The outputs are as follows:
*_img.png: Input image file.
*_pred_alb.png: The predicted albedo.
*_pred_shd.png: The predicted shading.

## Contact
If you have any questions, please contact P. Das.

## Citation
Please cite the paper if it is useful in your research:

```
@inproceedings{dasPIENet,
    title = {PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition}, 
    author = {Partha Das and Sezer Karaoglu and Theo Gevers},
              booktitle = {IEEE Conference on Computer Vision and Pattern Recognition, (CVPR)},
    year = {2022}
}
```
