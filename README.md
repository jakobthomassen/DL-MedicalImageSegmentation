# Candidate 27 and Candidate 16

Task 1: Impact of image resolution on the final outcome.
[Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8700246/ ]

Task 2: Impact of data augmentation techniques on the final predictions.
[Ref: https://imgaug.readthedocs.io/en/latest/index.html , https://albumentations.ai/ ]

Model: U-Net

Dataset: Kvasir-SEG

Kvasir-SEG is a publicly available colonoscopy image segmentation dataset comprising
1,000 polyp images with corresponding ground-truth masks. The dataset is compact (~46
MB) and well-suited for experimental studies. All annotations were manually delineated by a
medical expert and subsequently verified by an experienced gastroenterologist, ensuring
high-quality and clinically reliable segmentations.
The dataset is available at: https://datasets.simula.no/kvasir-seg/. In this project, students
are required to partition the dataset into 70% for training and 30% for validation when
conducting their experiments, in order to ensure consistency and comparability across
results.

train.txt contains the name of the training images and their corresponding masks.

val.txt contains the name of the validation images and their corresponding masks. 