# Machine Learning with PyTorch: A Practical Introduction

This repository contains a series of Jupyter notebooks that explore fundamental concepts in machine learning using PyTorch. The project covers a range of topics, from introductory examples like MNIST to more advanced applications in computer vision and natural language processing. The capstone of this project is an assessment that challenges the user to build a fruit classifier using transfer learning.

## Technologies and Concepts

This project provided hands-on experience with a variety of machine learning technologies and concepts, including:

*   **PyTorch:** The core deep learning framework used for building and training neural networks.
*   **Jupyter Notebooks:** An interactive environment for writing and running code, making it ideal for machine learning experimentation.
*   **Transfer Learning:** A key technique where a pre-trained model is adapted for a new task. In the assessment, the VGG16 model, pre-trained on the ImageNet dataset, was used as the base for the fruit classifier. This approach significantly reduces training time and improves performance, especially with smaller datasets.
*   **Convolutional Neural Networks (CNNs):** The architecture of choice for computer vision tasks. The notebooks demonstrate how to build, train, and fine-tune CNNs for image classification.
*   **Data Augmentation:** Techniques used to artificially expand the training dataset by creating modified copies of existing images. This helps to improve the model's ability to generalize to new, unseen data.
*   **Natural Language Processing (NLP):** The `06_nlp.ipynb` notebook provides an introduction to NLP, demonstrating how machine learning can be used to understand and process human language.

## The Assessment: Fresh vs. Rotten Fruit Classifier

The final assessment challenged me to build a model that could accurately classify images of fresh and rotten fruits. This task provided a practical application of the concepts learned throughout the project, with a particular focus on transfer learning.

### The Model

The solution involved using the VGG16 model as a feature extractor. The process was as follows:

1.  **Load the Pre-trained Model:** The VGG16 model, with weights pre-trained on the ImageNet dataset, was loaded.
2.  **Freeze the Base Layers:** To leverage the learned features of the VGG16 model, its convolutional layers were "frozen," meaning their weights were not updated during the initial training phase.
3.  **Add a Custom Classifier:** A new set of fully connected layers was added to the top of the VGG16 base. This new classifier was trained to distinguish between the six categories of fruit in the dataset (fresh apples, fresh bananas, fresh oranges, rotten apples, rotten bananas, and rotten oranges).
4.  **Train the Custom Classifier:** The model was then trained on the fruit dataset. During this phase, only the weights of the new, custom classifier were updated.
5.  **Fine-Tuning:** After the initial training, the entire model was "unfrozen," and the model was trained for a few more epochs with a very low learning rate. This process, known as fine-tuning, allows the pre-trained layers to adapt slightly to the specifics of the new dataset, often leading to a boost in performance.

### The Result

The model achieved an accuracy of **91.79%** on the validation set, which was just shy of the **92%** accuracy required to pass the assessment. This result demonstrates the power of transfer learning, as a highly accurate model was created with a relatively small dataset and limited training time. Further experimentation with data augmentation, hyperparameter tuning, and different model architectures could likely push the accuracy above the 92% threshold.

This project was a valuable learning experience that provided a solid foundation in both the theory and practice of machine learning.
