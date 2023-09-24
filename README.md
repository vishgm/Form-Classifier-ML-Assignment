#  Form Classifier - ML Assignment


![form_classifier_demo](https://github.com/vishgm/Form-Classifier-ML-Assignment/assets/54322539/badc593b-c442-4125-92f6-88d6c2baa123)


<br/>

**Demo:**  https://huggingface.co/spaces/vishgm/form-classifier-finetuned-vgg16

This solution is also deployed as a standalone webapp using Gradio (deployed on HuggingFace Spaces)


## Objective:
> To build a predictive model for classifying a scanned Form document into one of multiple form types (eg: Form X, Form D), if the document isn’t a form, then its classfied as category “Other”.


## Solution

**A complete analysis is available here : [form-classifier-notebook](https://github.com/vishgm/Form-Classifier-ML-Assignment/blob/main/jupyter-notebooks/Document_classification.ipynb)**

<a name="libraries"></a>
## Libraries:
```
Core libraries: numpy,sklearn,pandas,matplotlib
```

```
Machine Learning / NLP Libraries  : Sci-kit Learn(sk-learn), PyTesseract
```


```
Deep Learning Libraries : Tensorflow, Keras
```

## Summary 

To solve the above problem statement, following steps were performed:

- Download the original dataset (https://www.sec.gov/Archives/edgar/vprr/index.html), which consists of scanned form documents in PDF format.
- Segregate and label forms as per Form category (i.e: creating respective folders for each class: Form D, Form X, Form TA etc)
- Convert PDfs to image (here we only consider the first page present in each form)

It was also observed that there were less number of PDF files present for other classes except Form 13F which was the majority class.

To solve this, **data augmentation was performed on images of minority classes (added skew, noise etc)**

Next, we proceed with 2 solutions - NLP based and Deep Learning based (Image classification)

**Solution 1:** OCR using PyTesseract

**Solution 2:** Image classification using VGG16 model (using Tensorflow, keras)

## Result:
To determine the best model to use for this classification problem, a comparison was done between all of the models.

1) Solution 1: After performing **OCR using Tesseract**, it was found that since the nature of the documents are scans, it is difficult to get an accurate prediction using OCR and fuzzy matching, since OCR is highly dependant on Image quality. 

2) Solution 2:  **VGG16 model** finetuned on our custom dataset, provided the best results for category classification of Forms. This is because the solution is OCR-Free, and works well with scanned images since we use Transfer learning for document understanding on Image datasets (i.e we make it an image classification problem).  


## Data
The original dataset was found to be unstructured (Website with lots of pdfs). To better create a structured pipeline for model building and evaluation, the following was performed

- Downloaded the top 300 pdfs (consisting of different types of forms - Form 13F, Form D etc) from the website
- Created Train and Test folders
- Created Class-wise (category-wise folders) for storing PDFs of each class, for both train and test folders
- Converted PDFs to images and performed augmentations for the same for balancing records for each class / category 

Link: https://www.sec.gov/Archives/edgar/vprr/index.html

## @Authors

* **Vishak Gopkumar** - (https://github.com/vishgm)


