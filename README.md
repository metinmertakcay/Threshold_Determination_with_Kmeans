# Threshold_Determination_with_Kmeans
## HOW TO RUN PROGRAM?
- Go to directory where you download the project.
- Open console screen and enter “python main.py” command to run.

## REQUIREMENT
You can install necessary libraries using “pip install”.
- Matplotlib
- OpenCv
- Numpy

## WHERE SHOULD IMAGES BE REPLACED?
You have to put all input images in “bird images” folder because all image names are taken using “os” library.

## PROJECT STEP
- Image is read as grayscale.
- Gauss filter is applied to eliminate the noise in image.
- In order to determine threshold value, kmeans was applied to pixel values in the images.
- Pixel values lower than threshold have been changed 0 and pixel values higher than threshold have been changed to 255.
- Erosion which is a morphological analysis method, was applied in order to eliminate connections between adjacent birds.
- The connected component method was used to count the bird number.

## OUTPUT
[Click to see results.](/output)