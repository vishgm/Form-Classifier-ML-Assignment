"""
Description: Code for Document classification model - Gradio Webapp
Author: Vishak G.
Last modified: 23-09-2023
"""


import numpy as np
import gradio as gr
import uuid
import os

from doc_classifier_dl import VGG16DocumentClassifier

def predict_doc_type(input_img):

    req_id = str(uuid.uuid4())
    folder_save_path = f"./completed_requests/{req_id}"
    img_save_path = os.path.join(folder_save_path, "test.png")

    os.mkdir(folder_save_path)
    
    input_img.save(img_save_path, format="png")

    labels = ['82-Submission-sheet', 'Form-13F', 'Form-19B', 'Form-D', 'Form-TA', 'Form-X', 'Others']

    vgg_model = VGG16DocumentClassifier(train_path="../data/train", test_path=None, is_train=False, saved_model_path="../saved_models/Doc_classifier_vgg16")
    predictions = vgg_model.predict(img_path=img_save_path, classes=labels)

    return predictions


# demo = gr.Interface(
#     fn=predict_doc_type, 
#     inputs=gr.Image(type="pil"), 
#     outputs=gr.Label(num_top_classes=7),
#     examples = ["./examples/82_submission_sheet.png", "./examples/form_d.png"]
#     )

with gr.Blocks() as demo:

    gr.Markdown(
    """
    <h2 style="text-align: center;">Form Document classifer - ML Assignment</h2>
    <p style="text-align: center;"> This model was fine-tuned on custom dataset of Form documents using VGG16  - Author: Vishak Gopkumar</p>

    ---

    Steps to test model:

    - Upload an image by clicking the document uploader button below, or select on of the examples from the 'Examples' Sections
    - Click on predict
    - Check the results in the 'Label' tab - final result will display Form type (such as Form X, Form D, etc)
    
    """
    )

    with gr.Row():
        form_image = gr.Image(type="pil")
    
    with gr.Row():
        pred_result = gr.Label(num_top_classes=7)

    submit_btn = gr.Button("Predict")


    submit_btn.click(
        fn=predict_doc_type, 
        inputs=form_image,
        outputs=pred_result,
        # examples = ["./examples/82_submission_sheet.png", "./examples/form_d.png"]
    )

    gr.Examples(
        examples=["./examples/82_submission_sheet.png", "./examples/form_d.png"],
        inputs=form_image,
        outputs=pred_result,
        fn=predict_doc_type,
        # cache_examples=True
    )
    
if __name__ == "__main__":
    demo.launch()