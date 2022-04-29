# cs153-cinematic-feature-labeling

Final project for CS 153 Computer Vision: Classical Methods for Automated Feature Labeling in Cinematic Eye-Tracking Dataset.

## Installation Requirements

OpenCV Computer Vision Library:
`conda install -c conda-forge opencv`

Matplotlib for visualizing frames:
`conda install matplotlib`

FFmpeg for extracting frames from video clips:
`pip install imageio-ffmpeg`

## Data Setup

The notebooks in this repository assume that all 15 Gaze dataset clips are `mp4` files located in a `clips/` directory at the top level of this repository.

They also assume that the hand coding txt files for the dataset's ground-truth frame-by-frame annotations are located in a `hand_coding/` directory at the top level of the repository. These hand coding files can be obtained from [http://graphics.stanford.edu/~kbreeden/gazedata.html](http://graphics.stanford.edu/~kbreeden/gazedata.html).

The head localization notebooks also assume that there is a `frame_to_head_annotations.json` file from [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/), listing all 75 frames from the `frames_with_head_for_annotation/` directory used to evaluate head localization, as well as their hand-annotated head bounding box coordinates.

## Files

### ExtractGroundTruthLabels.ipynb

This notebook contains a script for extracting ground-truth face and cut labels from the hand coding provided in the Gaze dataset. Assumes that there is an existing `hand_coding/` directory containing one hand coding `txt` file for each clip.

Depending on what value `desired_feature` is set to (options: `"face"` or `"cut"`), the script will extract the ground-truth labels for either faces or cuts.

If `desired_feature` is set to `"face"`, this script produces `ground_truth_face_label_dicts/` directory containing a pickled dictionary for each clip, with the keys being the frame numbers and the values being either 0 or 1 indicating whether there was a ground-truth label for a face in the corresponding frame.

If `desired_feature` is set to `"cut"`, this script produces `ground_truth_cut_label_dicts/` directory containing a pickled dictionary for each clip, with the keys being the frame numbers and the values being either 0 or 1 indicating whether there was a ground-truth label for a cut in the corresponding frame.

### GenerateFrames.ipynb

This notebook contains a script for setting up the `frames/` directory structure to contain the frames from each clip. It also includes an example of the FFmpeg commands that should be run separately to generate the frames for each clip. These frames will be used in the rest of the notebooks, rather than the original video files themselves.

### HeadDetectionWithHaarCascades.ipynb

This notebook performs head detection and localization with Haar cascades, using the three Haar cascade xml files included in this repository, which can also be found at [https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades). Note that when downloading the xml files, it is important that the _raw_ versions be downloaded.

Assumes that there is an existing `frames/` directory.

Depending on what number `num_haar_cascades_to_use` is set to (options: `1`, `2`, or `3`), the script will run either 1, 2, or 3 Haar cascades.

For head detection, produces a pickled dictionary directory prefixed with `head_label_dicts_haar`. In that directory, there will be one dictionary for each clip, with the keys being the frame numbers and the values being either 0 or 1 indicating whether a head was detected in the corresponding frame.

For head localization, produces a head localization dictionary directory prefixed with `head_localization_dicts_haar`. That directory will contain one pickled dictionary for each clip, with the keys being the frame numbers and the values being lists of `(x,y,w,h)` tuples for the system-generated head bounding boxes for the corresponding frames.

Also produces a head localization image directory prefixed with `head_localization_haar`. In that directory, there will be one subfolder for each clip, with each subfolder containing all the frames for that clip with Haar cascade head bounding boxes drawn for each head detected.

### CutDetectionWithLKOpticalFlow.ipynb

This notebook performs cut detection with Lucas-Kanade optical flow. Assumes that there is an existing `frames/` directory.

Depending on what values are placed in the `err_thresholds` array and `num_votes_thresholds` array, the script will run Lucas-Kanade optical flow cut detection experiments with the specified `err_threshold` (flow error threshold) and `num_votes_threshold` (threshold for proportion of windows) combinations.

Produces `cut_label_dicts_lk_optical_flow/` directory containing one subfolder for each `err_threshold` and `num_votes_threshold` pair. Each subfolder contains a pickled dictionary for each clip, with the keys being the frame numbers and the values being either 0 or 1 indicating whether a cut was detected in the corresponding frame.

Also produces `lk_optical_flow_visualization/` directory containing one subfolder for each `err_threshold` and `num_votes_threshold` pair. Each subfolder contains one subfolder for each clip, with each inner subfolder containing all the frames for that clip with the tracks for Lucas-Kanade optical flow drawn onto each frame.

### EvaluateHeadDetectionAndCutDetection.ipynb

This notebook evaluates the results of our head detection and cut detection systems by computing accuracy, precision, recall, and F1 scores for each system.

To evaluate head detection, this script compares the pickled dictionaries for the head system labels (located in directories prefixed with `head_label_dicts_haar`) with the extracted face ground-truth labels (located in the `ground_truth_face_label_dicts/` directory). This comparison is done over all the clips in the dataset, and also broken down by clip.

To evaluate cut detection, this script compares the pickled dictionaries for the cut system labels (located in the `cut_label_dicts_lk_optical_flow/` directory) with the extracted cut ground-truth labels (located in the `ground_truth_cut_label_dicts/` directory). This comparison is done over all the clips in the dataset, and also broken down by clip. The script also plots a subset of the results from our `err_threshold` and `num_votes_threshold` experiments as precision-recall curves, saving those plots as `cut_detection_precision_recall_curve_lk_optical_flow_err_threshold.jpg` and `cut_detection_precision_recall_curve_lk_optical_flow_num_votes_threshold.jpg`.

### VisualizeHeadLocalizationResults.ipynb

This notebook provides a visualization function for head localization. This script loads in a head localization dictionary (e.g. a pickled dictionary from the directory `head_localization_dicts_haar`) with keys of frame numbers and values of lists of `(x,y,w,h)` tuples for the system-generated head bounding boxes for the corresponding frames. Using that information, this script is able to display the system-generated head bounding boxes for a particular film clip frame, and also to save those head localization results to a file called `head_localization_example_system.jpg`.

### ExtractAndVisualizeImageRegionsFromVIAJSONAnnotations.ipynb

This notebook extracts the head bounding box coordinates from VIA hand annotations saved a file called `frame_to_head_annotations.json` and saves them to a pickled dictionary called `frame_to_head_box_annotation.pkl` with keys of the frame filename and values of the `(x,y,w,h)` tuple for the head bounding box in the corresponding frame.

This notebook also includes code for visualizing the VIA hand-annotated head bounding box for a particular frame and saving that visualization to a file called `head_localization_example_ground_truth_annotation.jpg`.

### EvaluateHeadLocalization.ipynb

This notebook evaluates the results of our head localization systems by computing the mean Intersection over Union (IoU) score for each system. It compares the pickled dictionaries for the head bounding box system labels (located in directories prefixed with `head_localization_dicts_haar`) with the extracted head bounding box ground-truth coordinates annotated with VIA (located in `frame_to_head_box_annotation.pkl`), and reports the mean IoU score for all 75 hand-annotated frames, the number of frames where a head was detected, and the mean IoU score over just the frames where a head was detected.

This notebook also includes code for simultaneously visualizing the VIA hand-annotated head bounding box and the system-generated head bounding box for a particular frame and saving that visualization to a file called `head_localization_example_evaluation.jpg`.
