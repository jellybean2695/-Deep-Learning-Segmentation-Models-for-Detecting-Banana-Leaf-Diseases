# -Deep-Learning-Segmentation-Models-for-Detecting-Banana-Leaf-Diseases
The timely and accurate detection of banana leaf diseases is essential for reducing agricultural losses and promoting sustainable farming practices. This study begins with the development of a robust preprocessing pipeline, incorporating techniques such as image resizing, normalization, and data augmentation methods like flipping, rotation, and scaling, to enhance the diversity and quality of training data. Leveraging a dataset of annotated banana leaf images, the project evaluates and compares the performance of two deep learning segmentation architectures: U-Net and SegNet. Both models are trained to detect and segment diseased regions, with their outputs assessed using metrics such as accuracy, IoU, Dice coefficient, precision, and recall. U-Net excels in capturing fine-grained features, while SegNet offers computational efficiency, making it suitable for resource-constrained environments. Visual comparisons of segmentation maps and performance graphs provide actionable insights into model selection for real-world agricultural disease detection systems, contributing to the advancement of automated solutions in precision agriculture.
## Project Workflow

1. `preprocess.py` - Data preprocessing
2. `generate_masks.py` - Create ground truth masks
3. `unet_model.py` - Define U-Net model
4. `segnet_model.py` - Define SegNet model
5. `train_models.py` - Train the models
6. `evaluate_models.py` - Evaluate performance
7. `visualize_predictions.py` - Visualize model outputs
