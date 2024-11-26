# RUOK: Recuration of a Public Dataset Utilized to Optimize Knowledge for Multi-label Chest X-ray Disease Screening (Accepted: presentation)

[![Python 3.8](https://img.shields.io/badge/Python-3.8.13-3776AB.svg?style=flat&logo=python&logoColor=yellow)](https://www.python.org/downloads/release/python-3813/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.1+cu118-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![PyTorch Lightning 2.1](https://img.shields.io/badge/pytorch-lightning-792ee5.svg?logo=PyTorch%20Lightning)](https://lightning.ai/pytorch-lightning)
![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)
[![Apache License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)]()
![ONNX](https://a11ybadges.com/badge?logo=onnx)

PyTorch/Onnx Implementation of RUOK: Recuration of a Public Dataset Utilized to Optimize Knowledge for Multi-label Chest X-ray Disease Screening, presented at the EMBC 2024 conference.

# Abstract

RUOK, a deep learning model has been implemented to support the screening protocol at hospitals located in rural areas of Thailand, addressing a shortage of radiologists in the region. Although various publicly available Chest X-ray (CXR) datasets exist, their labels primarily focus on radiological abnormalities rather than diseases, limiting their suitability for training disease screening models. To address this limitation, we collaborated with expert radiologists to reclassify the labels of public datasets into eight classes representing prevalent diseases in Thailand: No finding, Suspected active tuberculosis, Suspected lung malignancy, Abnormal heart, great vessels, and mediastinum, Intrathoracic abnormal findings, Pneumonia, COVID-19, and Extrathoracic abnormal findings. This innovative adaptation of public dataset labels to our domain enhances the model's optimization for practical screening protocols. The data-efficient image transformers model was utilized to classify common diseases in Thailand using CXR images sourced from public datasets. The Area Under the Receiver Operating Characteristics (AUROC) evaluated on test set yielded the following results: 93.06%, 89.21%, 68.66%, 90.46%, 66.88%, 73.45%, 75.13%, and 80.63% for its respective classes. This approach highlights the potential of our model in advancing disease classification and improving healthcare outcomes, especially in regions facing a shortage of radiological expertise.

---


<img src="https://github.com/NailEm-CoE/RUOK/assets/15160408/fd5b2fb5-805b-4d0d-9ced-0c010d5f7e1f" alt="ruok-diagram" height="300">

---

# Installation

You can install the dependencies using this command:

```bash
git clone https://github.com/NailEm-CoE/RUOK
cd RUOK
pp instsall -r requirements.txt
```

For details, RUOK was implemented and trained with Python `3.8.13` and [Pytorch](https://pytorch.org/) `2.1.1+cu118` using the [Pytorch Lightning framework](https://lightning.ai/pytorch-lightning) version `2.1.3`. However, for this repository, you only need PyTorch.

# Inference

![ruok-heatmap-1](https://github.com/NailEm-CoE/RUOK/assets/15160408/a90279c1-fd24-4165-8c7b-7cc94bfac0c1)

To run the model, please refer to the [example.ipynb](example.ipynb) notebook for detailed instructions.

Alternatively, you can use the Replicate Playground for inference by visiting the following URL: [https://replicate.com/tu-cils/ruok_v1](https://replicate.com/tu-cils/ruok_v1).

# Citation

Please cite our paper if you find this code or this work is useful:

```bibtex
@inproceedings{ruok,
  title={RUOK: Recuration of a Public Dataset Utilized to Optimize Knowledge for Multi-label Chest X-ray Disease Screening},
  author={P. Boonyarattanasoontorn and P. Tieanworn and S. Puangarom and others},
  booktitle={2024 46th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={},
  year={2024},
  organization={IEEE}
}
```

# Reference

```bibtex
AIChest4All @inproceedings{thammarach2020ai,
  title={AI chest 4 all},
  author={Thammarach, Purinat and Khaengthanyakan, Suntara and Vongsurakrai, Sethavudh and Phienphanich, Phongphan and Pooprasert, Pakinee and Yaemsuk, Akarachai and Vanichvarodom, Podsirin and Munpolsri, Namtip and Khwayotha, Sirihattaya and Lertkowit, Meyhininat and others},
  booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={1229--1233},
  year={2020},
  organization={IEEE}
}
```
