# Playing Card Classification using PyTorch

## Overview

This project implements a convolutional neural network (CNN) in PyTorch to classify playing cards. The model is trained on a dataset of card images, categorized into different classes based on suit and rank. The model predicts the class of a given playing card image.

## Requirements

Install the required dependencies before running the project:

```bash
pip install torch torchvision pillow
```

Install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of images of playing cards, structured in subdirectories where each folder represents a card category. The dataset is loaded using `torchvision.datasets.ImageFolder`, and images are preprocessed with transformations like resizing, normalization, and tensor conversion.

## Project Structure

```
|-- project_directory/
    |-- Cards_dataset/      # Dataset containing training and testing images
        |-- train/          # Training images organized in subdirectories by class
        |-- test/           # Testing images organized similarly
    |-- main.py             # Main script for training and evaluation
    |-- model.py            # Defines the CNN model architecture
    |-- README.md           # Project documentation
    |-- requirements.txt    # List of dependencies
```

## Code Components

- **Data Loading**: Uses `torchvision.datasets.ImageFolder` to load playing card images and applies transformations.
- **Transformations**: Images are resized, normalized, and converted to tensors.
- **Custom Dataset**: A `CustomImageFolder` class assigns labels based on predefined card categories.
- **Neural Network**: A simple CNN is defined using `torch.nn.Module` with convolutional and fully connected layers.
- **Training**: The model is trained using `torch.optim.Adam` with `CrossEntropyLoss`.
- **Evaluation**: The trained model is tested for accuracy on unseen card images.
- **Prediction**: A single image can be classified using the trained model.

## Usage
Modify the path with your file path
Download cards dataset from kaggle

1. Train the model by running:

```bash
python main.py
```

2. Test the model to evaluate accuracy.

3. Use the trained model to classify a new playing card image.


## License

This project is open-source and can be modified or distributed under the MIT License.

