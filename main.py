import nncf
import torch

import numpy as np
import torch
from sklearn.metrics import accuracy_score

import openvino as ov

# Step1: Prepare calibration and validation datasets
calibration_loader = torch.utils.data.DataLoader(...)

def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
validation_dataset = nncf.Dataset(calibration_loader, transform_fn)

# Step2: Prepare validation function
def validate(model: ov.CompiledModel, 
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)

# Step3: Run quantization with accuracy control
model = ... # openvino.Model object

quantized_model = nncf.quantize_with_accuracy_control(
    model,
    calibration_dataset=calibration_dataset,
    validation_dataset=validation_dataset,
    validation_fn=validate,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE,
)

# Step4: Save quantized model
ov.save_model(quantized_model, "quantized_model.xml", compress_to_fp16=False)
