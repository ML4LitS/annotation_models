# Accelerating Biomedical Named Entity Recognition with Quantised EPMCA Bioformer-8L (QEB8L) Model

Santosh Tirunagari, Melissa Harrison

---
## Summary
we present the Quantised EPMCA Bioformer-8L (QEB8L) model for Biomedical Named Entity Recognition. By utilizing the Onnx runtime and quantisation techniques, we achieved a faster and lighter model without compromising performance. The results demonstrate comparable performance to Biobert but with a significant speed improvement.

## Introduction
Biomedical Named Entity Recognition (NER) poses a significant challenge in biomedical information processing due to the extensive lexical variations and ambiguity of out-of-context terms. Recent advancements in models such as BERT, GPT, and LLMs have shown improved performance on bioNER benchmarks. However, these models often demand substantial computational resources for production. We introduce the quantised_epmca_bioformer-8L (QEB8L) model, trained on the Europe PMC fully annotated corpus for genes/proteins, diseases, chemicals, and organisms. The QEB8L model leverages the ONNX runtime and is quantised, resulting in a lighter (77MB) and faster inference process. It achieves comparable results to Biobert but exhibits a remarkable 10x speed improvement on 2-core CPU machines with 1 GB RAM.

A comprehensive, step-by-step guide for running the QEB8L model and setting up the required environment.

## Installation and Environment Setup

To utilize the QEB8L model for Biomedical Named Entity Recognition, follow the steps below to install Python3, Pip3, and create a virtual environment:

### Mac and Linux

1. Install Python3:
   - Open the terminal.
   - Update package lists: `sudo apt update`.
   - Install Python3: `sudo apt install python3`.

2. Install Pip3:
   - Run: `sudo apt install python3-pip`.

3. Install Virtualenv:
   - Execute: `pip3 install virtualenv`.

4. Create a virtual environment:
   - Navigate to the desired directory in the terminal.
   - Run: `virtualenv myenv`.

5. Activate the virtual environment:
   - In the terminal, navigate to the virtual environment's directory.
   - Execute: `source myenv/bin/activate`.

6. Install the required Python packages:
   - In the activated virtual environment, run the following commands to install the required packages:


```
pip install optimum==1.16.2
pip install onnx==1.13.1
pip install onnxruntime==1.15.1
```


## Model Loading and Usage
Once the virtual environment is activated, follow these steps to load and utilise the QEB8L model:

Download/clone the model from the repo and specify the path to the downloaded model folder by creating a variable in Python 

```
quantised_path = folder_where_the_model_is_located
```

1. Import the required libraries:

```
from optimum.pipelines import pipeline
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
```

2. Load the quantized model and tokenizer:

```
model_quantized = ORTModelForTokenClassification.from_pretrained(quantised_path, file_name="model_quantized.onnx")
tokenizer_quantized = AutoTokenizer.from_pretrained(quantised_path, model_max_length=512, batch_size=8, truncation=True)
```

3. Create a pipeline for token classification:

```
ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized, aggregation_strategy="first")
```

4. Provide a sample text for Named Entity Recognition:

```
text = '''CLASS Omicron is a variant of SARS-CoV-2 first reported to the World Health Organization by the Network for Genomics Surveillance in South Africa on 24 November 2021. It was first detected in Botswana and has spread to become the predominant variant in circulation around the world.. The SARS-CoV-2 uses ACE2 to infect target cells and the expression of the ACE2 levels are increased following treatment with angiotensin-converting enzyme inhibitors (ACEIs); in addition, angiotensin receptor blockers (ARBs) has emerged speculation that patients with COVID-19 receiving these drugs may be under at a potentially increased risk for developing severe and fatal illness [11, 12]. Interestingly, when looking at ACE2 expression in different pathological stages (1 to 4), no differences was observed in any of the two lung cancer types (Figure 1C-D), suggesting stage might not the factor affecting ACE2 expression in lung tumor and therefore no significant differences in the susceptibility to SARS-CoV-2 infection among the pathological stages for LUAD and LUSC patients. Chronic kidney disease (CKD) is a global public health problem, and its prevalence is gradually increasing, mainly due to an increase in the number of patients with type 2 diabetes mellitus (T2DM) [1,2,3,4].  Human multidrug and toxin extrusion member 2 (MATE2-K, SLC47A2) plays an important role in the renal elimination of various clinical drugs including the antidiabetic drug metformin. The goal of this study was to characterize genetic variants of MATE2-K and determine their association with the pharmacokinetics of metformin'''
```

5. Perform Named Entity Recognition:
```
pred = ner_quantized(text)
```

6. Visualize the extracted entities:

```
for ent in pred:
    print([ent['start'], ent['end'], text[ent['start']:ent['end']], ent['entity_group'], ent['score']])
```

The output is listed in the following format: [start_span,end_span,entity,entity_type,score].

The entity types are as follows: 'GP': Gene/Protein, 'CD': Chemical/Drug, 'OG': Organism, and 'DS': Disease. This format allows you to identify the start and end positions of the entity in the text, the entity itself, its corresponding entity type, and the associated score.

```
[30, 40, 'SARS-CoV-2', 'OG', 0.98088056]
[288, 298, 'SARS-CoV-2', 'OG', 0.9897682]
[304, 308, 'ACE2', 'GP', 0.9994]
[358, 362, 'ACE2', 'GP', 0.9993819]
[409, 438, 'angiotensin-converting enzyme', 'GP', 0.9984209]
[472, 492, 'angiotensin receptor', 'GP', 0.99874496]
[552, 560, 'COVID-19', 'DS', 0.99102575]
[709, 713, 'ACE2', 'GP', 0.9994085]
[814, 825, 'lung cancer', 'DS', 0.9988202]
[895, 899, 'ACE2', 'GP', 0.9993438]
[914, 924, 'lung tumor', 'DS', 0.9986173]
[991, 1011, 'SARS-CoV-2 infection', 'DS', 0.9965901]
[1046, 1050, 'LUAD', 'DS', 0.9971084]
[1055, 1059, 'LUSC', 'DS', 0.99671656]
[1070, 1092, 'Chronic kidney disease', 'DS', 0.998764]
[1094, 1097, 'CKD', 'DS', 0.9988304]
[1235, 1259, 'type 2 diabetes mellitus', 'DS', 0.99851716]
[1261, 1265, 'T2DM', 'DS', 0.9988927]
[1279, 1284, 'Human', 'OG', 0.99373156]
[1285, 1323, 'multidrug and toxin extrusion member 2', 'GP', 0.99628574]
[1325, 1332, 'MATE2-K', 'GP', 0.99739236]
[1334, 1341, 'SLC47A2', 'GP', 0.9994468]
[1450, 1459, 'metformin', 'CD', 0.99891806]
[1524, 1531, 'MATE2-K', 'GP', 0.99751383]
[1593, 1602, 'metformin', 'CD', 0.9987571]
```

## Performance
| Metric     | BioBert (CD) | QEB8L (CD) | BioBert (DS) | QEB8L (DS) | BioBert (OG) | QEB8L (OG) | BioBert (GP) | QEB8L (GP) |
|------------|--------------|------------|--------------|------------|--------------|------------|--------------|------------|
| Precision  | 0.91         | 0.85       | 0.90         | 0.90       | 0.93         | 0.94       | 0.91         | 0.90       |
| Recall     | 0.92         | 0.90       | 0.80         | 0.88       | 0.86         | 0.85       | 0.87         | 0.88       |
| F1 Score   | 0.92         | 0.88       | 0.85         | 0.89       | 0.90         | 0.89       | 0.89         | 0.89       |


## Parameters selected
| Parameter                        | Value                              | Description                                                |
|----------------------------------|------------------------------------|------------------------------------------------------------|
| `evaluation_strategy`            | `"epoch"`                          | Evaluate the model at the end of each epoch                |
| `learning_rate`                  | `1e-5`                             | Learning rate for training                                 |
| `num_train_epochs`               | `5`                                | Number of training epochs                                  |
| `weight_decay`                   | `0.01`                             | Weight decay for regularization                            |
| `warmup_steps`                   | `500`                              | Number of warmup steps during training                     |
| `metric_for_best_model`          | `'f1'`                             | Metric used to determine the best model                    |
| `greater_is_better`              | `True`                             | Whether a higher value for the metric is better            |
| `gradient_accumulation_steps`    | `2`                                | Number of steps to accumulate gradients before updating    |


## Cite 
1. APA 
<pre>
Tirunagari, S., & Harisson, M. (2023). Accelerating Biomedical Named Entity Recognition with Quantised EPMCA Bioformer-8L (QEB8L) Model (Version 0.0.1) [Computer software]. Retrieved from [https://github.com/ML4LitS/annotation_models](https://github.com/ML4LitS/annotation_models)
</pre>

2. Bibtex
<pre>
@misc{Tirunagari_Harisson_2023,
  author       = {Santosh Tirunagari and Melissa Harisson},
  title        = {Accelerating Biomedical Named Entity Recognition with Quantised EPMCA Bioformer-8L (QEB8L) Model},
  year         = {2023},
  version      = {0.0.1},
  url          = {https://gitlab.ebi.ac.uk/literature-services/machine-learning/qeb8l},
  note         = {Software available from \url{https://gitlab.ebi.ac.uk/literature-services/machine-learning/qeb8l}},
  howpublished = {GitLab repository},
  month        = jun,
  orcid        = {0000-0002-9064-1965}
}
</pre>

## Licence
CC-by


