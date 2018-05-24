# Instance Segmentation

Beyond image classification and object detection, the Computer Vision problems where a single thing must be recognized or multiple class instances should be recognized and detected eith bounding boxes, a new challenge appears: Semantic and Instance Segmentation.  
The main objective of this approach is to recognize pixel-wise to which class belongs every instance of a given image, supposing that the model we're applying's been trained on this classes. This project uses the TensorFlow object detection Mask RCNN model to achieve that. Several scripts are provided depending on what functionality is required.

## Dependencies



## Use

Simply activate the virtual environment and execute the Python scripts.

### instance_segmentation.py

This program will take the input image file, feed it into the Segmentation model and generate two windows with the original and the processed image. Use example:

```
python instance_segmentation.py -i images/my_image.jpg
```

By default, if not images are provided, this output'll be displayed.

![instance_segmented_image](segmented_images/scotty.jpg "instance_segmented_image")

### instance_segmentation_images.py

The script will process every *.jpg* image in the *images/* folder and save them in the *segmented_images/* folder.

### instance_segmentation_webcam.py

Here, the instance segmentation'll be applied to the frame captured by the first available webcam. Press the 'q' button in the generated floating windows or *Ctrl+C* to stop the execution. Note that this is a high demanding task in computational terms, even when using the most lightweight model available. Consider using GPU acceleration for this purposes.
