# MTCNN-v2 face detector (mxnet version)

Author: zhaoyafei0210@gmail.com (https://github.com/walkoncross)

This repo is based on repo: https://github.com/pangyupo/mxnet_mtcnn_face_detection.

MtcnnDetector in mtcnn_detetctor.py is an class wrapper for the implementation of MTCNN-v2 which has 4 stages: PNet, RNet, ONet and LNet.

Except MtcnnDetector, another 2 class wrappers: MtcnnAligner (in mtcnn_aligner.py) and FaceAligner (in face_aligner.py) are provided for easy usage.

## Contents
```
├───face_aligner            // mtcnn wrapped as a class 'FaceAligner', which is a super-class of 'MtcnnAligner'. It loads an image and a list of face rects, detects landmarks, does alignment and cropping, outputs aligned face chips.
├───model                   // mtcnn's mxnet model
├───mtcnn_aligner           // mtcnn warpped as a class 'MtcnnAligner' with 3 stages (RNet, ONet, LNet), PNet removed. It loads an image and a list of face rects, and detects 5 landmarks for each rects, output a list of 5 landmarks.
├───mtcnn_detector          // mtcnn warpped as a class 'MtcnnDetector' with 4 stages (PNet, RNet, ONet, LNet).
├───scripts                 // some self-used test scripts
├───test_imgs               // test images
```
