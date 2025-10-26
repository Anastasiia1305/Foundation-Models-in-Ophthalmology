[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](#)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](#)

# Awesome Ophthalmic Foundation Models 👁️

This repository contains a curated list of foundation models for ophthalmic imaging and retinal analysis across various imaging modalities (e.g., CFP, OCT). 
This list focuses specifically on vision and vision–language models trained on ophthalmic imaging data. Large language models (LLMs) without an imaging component are intentionally excluded.
For each model, we provide links to the paper and available code, along with key model details. The goal is to support transparency, comparability, and reproducibility by making it easier to see what exists, how it was trained, and how it can be used. 

💡 **This list is continuously updated. Contributions welcome!**

---

## Table of Contents
- [What is this?](#what-is-this)
- [How to Cite](#how-to-cite)
- [Models](#models)
  - [2025](#2025)
  - [2024](#2024)
  - [2023](#2023)
- [License](#license)

---

## What is this?

A foundation model is a large neural network pretrained on massive, diverse data, so it learns general-purpose features instead of just solving one task.

In ophthalmology, these models are trained on millions of retinal and ocular images (CFP, OCT, FFA, etc.) and sometimes paired with clinical text. After pretraining, the same model can be adapted to many downstream tasks (e.g., diagnosis, segmentation, report generation) with minimal extra data, and in some cases, the pretrained weights are released for reuse.

This repository is intended to serve as a living index of these models.

---

## How to Cite 

If you use this list in your work, please consider citing this paper: [DETAILS]

---

## Models

Each model entry contains:
- links to paper and code,
- primary imaging modalities used for pretraining,
- an expandable abstract with details.

---

### 2025

**Training a high-performance retinal foundation model with half-the-data and 400 times less compute (RETFound-Green)**  
[📄 Paper](https://www.nature.com/articles/s41467-025-62123-z) | [💻 Code](https://github.com/justinengelmann/RETFound_Green)  
_Modality:_ CFP

<details>
<summary><strong>Abstract</strong></summary>

Medical artificial intelligence is limited by available training datasets. Foundation models like RETFound from Moorfields Eye Hospital (MEH) can be adapted with small downstream datasets and thus alleviate this issue. RETFound-MEH used 900,000 training images. Recently, “data-efficient” DERETFound achieved comparable performance with 150,000 images. Both require very substantial compute resources for training and use. We propose RETFound-Green trained on only 75,000 publicly available images with 400 times less compute using a novel Token Reconstruction objective. RETFound-MEH and DERETFound training costs are estimated at $10,000 and $14,000, respectively. RETFound-Green cost less than $100, with equally reduced environmental impact. RETFound-Green can be downloaded 14 times faster, computes vector embeddings 2.7 times faster which then require 2.6 times less storage space. On a variety of downstream tasks from geographically diverse datasets, RETFound-Green achieves more than twice as many statistically significant wins than the next best model.

</details>

---

**A multimodal visual–language foundation model for computational ophthalmology (EyeCLIP)**  
[📄 Paper](https://www.nature.com/articles/s41746-025-01772-2) | [💻 Code](https://github.com/Michi-3000/EyeCLIP)  
_Modalities:_ CFP, FFA, Slit-lamp, ICGA, OUS, OCT, Specular microscope, FAF, External eye photo, Corneal topography, RetCam

<details>
<summary><strong>Abstract</strong></summary>

Early detection of eye diseases is vital for preventing vision loss. Existing ophthalmic artificial intelligence models focus on single modalities, overlooking multi-view information and struggling with rare diseases due to long-tail distributions. We propose EyeCLIP, a multimodal visual-language foundation model trained on 2.77 million ophthalmology images from 11 modalities with partial clinical text. Our novel pretraining strategy combines self-supervised reconstruction, multimodal image contrastive learning, and image-text contrastive learning to capture shared representations across modalities. EyeCLIP demonstrates robust performance across 14 benchmark datasets, excelling in disease classification, visual question answering, and cross-modal retrieval. It also exhibits strong few-shot and zero-shot capabilities, enabling accurate predictions in real-world, long-tail scenarios. EyeCLIP offers significant potential for detecting both ocular and systemic diseases, and bridging gaps in real-world clinical applications.

</details>

---

**VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge (VisionUnite)**  
[📄 Paper](https://arxiv.org/abs/2408.02865) | [💻 Code](https://github.com/HUANGLIZI/VisionUnite)  
_Modality:_ CFP

<details>
<summary><strong>Abstract</strong></summary>

The need for improved diagnostic methods in ophthalmology is acute, especially in the underdeveloped regions with limited access to specialists and advanced equipment. Therefore, we introduce VisionUnite, a novel vision-language foundation model for ophthalmology enhanced with clinical knowledge. VisionUnite has been pretrained on an extensive dataset comprising 1.24 million image-text pairs, and further refined using our proposed MMFundus dataset, which includes 296,379 high-quality fundus image-text pairs and 889,137 simulated doctor-patient dialogue instances. Our experiments indicate that VisionUnite outperforms existing generative foundation models such as GPT-4V and Gemini Pro. It also demonstrates diagnostic capabilities comparable to junior ophthalmologists. VisionUnite performs well in various clinical scenarios including open-ended multi-disease diagnosis, clinical explanation, and patient interaction, making it a highly versatile tool for initial ophthalmic disease screening. VisionUnite can also serve as an educational aid for junior ophthalmologists, accelerating their acquisition of knowledge regarding both common and underrepresented ophthalmic conditions. VisionUnite represents a significant advancement in ophthalmology, with broad implications for diagnostics, medical education, and understanding of disease mechanisms. 

</details>

---

**Specialized curricula for training vision language models in retinal image analysis (RetinaVLM-Specialist)**  
[📄 Paper](https://www.nature.com/articles/s41746-025-01893-8) | [💻 Code](https://github.com/RobbieHolland/SpecialistVLMs)  
_Modality:_ OCT

<details>
<summary><strong>Abstract</strong></summary>

Clinicians spend significant time reviewing medical images and transcribing findings. By integrating visual and textual data, foundation models have the potential to reduce workloads and boost efficiency, yet their practical clinical value remains uncertain. In this study, we find that OpenAI’s ChatGPT-4o and two medical vision-language models (VLMs) significantly underperform ophthalmologists in key tasks for age-related macular degeneration (AMD). To address this, we developed a dedicated training curriculum, designed by domain specialists, to optimize VLMs for tasks related to clinical decision making. The resulting model, RetinaVLM-Specialist, significantly outperforms foundation medical VLMs and ChatGPT-4o in AMD disease staging (F1: 0.63 vs. 0.33) and referral (0.67 vs. 0.50), achieving performance comparable to junior ophthalmologists. In a reader study, two senior ophthalmologists confirmed that RetinaVLM’s reports were substantially more accurate than those written by ChatGPT-4o (64.3% vs. 14.3%). Overall, our curriculum-based approach offers a blueprint for adapting foundation models to real-world medical applications.

</details>

---

**A Foundation Language-Image Model of the Retina (FLAIR): encoding expert knowledge in text supervision (FLAIR)**  
[📄 Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002822) | [💻 Code](https://github.com/jusiro/FLAIR)  
_Modality:_ CFP

<details>
<summary><strong>Abstract</strong></summary>

Foundation vision-language models are currently transforming computer vision, and are on the rise in medical imaging fueled by their very promising generalization capabilities. However, the initial attempts to transfer this new paradigm to medical imaging have shown less impressive performances than those observed in other domains, due to the significant domain shift and the complex, expert domain knowledge inherent to medical-imaging tasks. Motivated by the need for domain-expert foundation models, we present FLAIR, a pre-trained vision-language model for universal retinal fundus image understanding. To this end, we compiled 38 open-access, mostly categorical fundus imaging datasets from various sources, with up to 101 different target conditions and 288,307 images. We integrate the expert’s domain knowledge in the form of descriptive textual prompts, during both pre-training and zero-shot inference, enhancing the less-informative categorical supervision of the data. Such a textual expert’s knowledge, which we compiled from the relevant clinical literature and community standards, describes the fine-grained features of the pathologies as well as the hierarchies and dependencies between them. We report comprehensive evaluations, which illustrate the benefit of integrating expert knowledge and the strong generalization capabilities of FLAIR under difficult scenarios with domain shifts or unseen categories. When adapted with a lightweight linear probe, FLAIR outperforms fully-trained, dataset-focused models, more so in the few-shot regimes. Interestingly, FLAIR outperforms by a wide margin larger-scale generalist image-language models and retina domain-specific self-supervised networks, which emphasizes the potential of embedding experts’ domain knowledge and the limitations of generalist models in medical imaging. 

</details>

---

### 2024

**Expertise-informed Generative AI Enables Ultra-High Data Efficiency for Building Generalist Medical Foundation Model (DERETFound)**  
[📄 Paper](https://www.researchsquare.com/article/rs-3766549/v1) | [💻 Code](https://github.com/Jonlysun/DERETFound)  
_Modality:_ CFP

<details>
<summary><strong>Abstract</strong></summary>

Generalist medical foundation models, pre-trained on massive medical datasets, have shown great potential as the next generation of medical artificial intelligence (AI). However, collecting millions of medical data is extremely expensive, time-consuming, and raises concerns over the high-risk leakage of sensitive private patient information. Here, we present a general framework that enables ultra-high data efficiency in building medical foundation models by leveraging expertise-informed generative AI to scale the limited pre-training dataset. Specifically, we follow this framework and propose a new foundation model DERETFound in ophthalmology, using only 16.7% (150,786 images) of the real-world colour fundus photography images required in the latest retinal foundation model RETFound (904,170 images, Y. Zhou et al, Nature 2023). By integrating expert insights into generative AI, we generate approximately one million synthetic data that are consistent with real retinal images in terms of physiological structures and feature distribution. DERETFound achieves comparable or even superior performance to RETFound on nine public datasets across four downstream tasks, including diabetic retinopathy grading, glaucoma diagnosis, age-related macular degeneration grading, multi-disease classification, and challenging external evaluation. In addition, DERETFound demonstrates competitively high label efficiency, saving over 50% of expert-annotated training data compared to RETFound on datasets for diabetic retinopathy grading. Our data-efficient framework challenges the classic view that building medical foundation models requires the collection of large amounts of real-world medical data as a prerequisite. The framework also provides an effective solution for any other diseases that were once discouraged from building foundation models due to limited data, which has profound significance for medical AI.

</details>

---

**EyeFound: A Multimodal Generalist Foundation Model for Ophthalmic Imaging (EyeFound)**  
[📄 Paper](https://arxiv.org/abs/2405.11338) | [💻 Code]() 🔒 Code not publicly available  
_Modalities:_ CFP, FFA, ICGA, FAF, RetCam, OUS, OCT, Slit-lamp, External eye photo, Specular microscope, Corneal topography

<details>
<summary><strong>Abstract</strong></summary>

Artificial intelligence (AI) is vital in ophthalmology, tackling tasks like diagnosis, classification, and visual question answering (VQA). However, existing AI models in this domain often require extensive annotation and are task-specific, limiting their clinical utility. While recent developments have brought about foundation models for ophthalmology, they are limited by the need to train separate weights for each imaging modality, preventing a comprehensive representation of multi-modal features. This highlights the need for versatile foundation models capable of handling various tasks and modalities in ophthalmology. To address this gap, we present EyeFound, a multimodal foundation model for ophthalmic images. Unlike existing models, EyeFound learns generalizable representations from unlabeled multimodal retinal images, enabling efficient model adaptation across multiple applications. Trained on 2.78 million images from 227 hospitals across 11 ophthalmic modalities, EyeFound facilitates generalist representations and diverse multimodal downstream tasks, even for detecting challenging rare diseases. It outperforms previous work RETFound in diagnosing eye diseases, predicting systemic disease incidents, and zero-shot multimodal VQA. EyeFound provides a generalizable solution to improve model performance and lessen the annotation burden on experts, facilitating widespread clinical AI applications for retinal imaging.

</details>

---

**UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling (UrFound)**  
[📄 Paper](https://arxiv.org/abs/2408.05618) | [💻 Code](https://github.com/yukkai/UrFound)  
_Modalities:_ CFP, OCT

<details>
<summary><strong>Abstract</strong></summary>

Retinal foundation models aim to learn generalizable representations from diverse retinal images, facilitating label-efficient model adaptation across various ophthalmic tasks. Despite their success, current retinal foundation models are generally restricted to a single imaging modality, such as Color Fundus Photography (CFP) or Optical Coherence Tomography (OCT), limiting their versatility. Moreover, these models may struggle to fully leverage expert annotations and overlook the valuable domain knowledge essential for domain-specific representation learning. To overcome these limitations, we introduce UrFound, a retinal foundation model designed to learn universal representations from both multimodal retinal images and domain knowledge. UrFound is equipped with a modality-agnostic image encoder and accepts either CFP or OCT images as inputs. To integrate domain knowledge into representation learning, we encode expert annotation in text supervision and propose a knowledge-guided masked modeling strategy for model pre-training. It involves reconstructing randomly masked patches of retinal images while predicting masked text tokens conditioned on the corresponding retinal image. This approach aligns multimodal images and textual expert annotations within a unified latent space, facilitating generalizable and domain-specific representation learning. Experimental results demonstrate that UrFound exhibits strong generalization ability and data efficiency when adapting to various tasks in retinal image analysis. By training on ~180k retinal images, UrFound significantly outperforms the state-of-the-art retinal foundation model trained on up to 1.6 million unlabelled images across 8 public retinal datasets. 

</details>

---

**RET-CLIP: A Retinal Image Foundation Model Pre-trained with Clinical Diagnostic Reports (RET-CLIP)**  
[📄 Paper](https://arxiv.org/abs/2405.14137) | [💻 Code](https://github.com/sStonemason/RET-CLIP)  
_Modality:_ CFP

<details>
<summary><strong>Abstract</strong></summary>

The Vision-Language Foundation model is increasingly investigated in the fields of computer vision and natural language processing, yet its exploration in ophthalmology and broader medical applications remains limited. The challenge is the lack of labeled data for the training of foundation model. To handle this issue, a CLIP-style retinal image foundation model is developed in this paper. Our foundation model, RET-CLIP, is specifically trained on a dataset of 193,865 patients to extract general features of color fundus photographs (CFPs), employing a tripartite optimization strategy to focus on left eye, right eye, and patient level to reflect real-world clinical scenarios. Extensive experiments demonstrate that RET-CLIP outperforms existing benchmarks across eight diverse datasets spanning four critical diagnostic categories: diabetic retinopathy, glaucoma, multiple disease diagnosis, and multi-label classification of multiple diseases, which demonstrate the performance and generality of our foundation model. 

</details>

---

**OCTCube-M: A 3D multimodal optical coherence tomography foundation model for retinal and systemic diseases with cross-cohort and cross-device validation (OCTCube-M)**  
[📄 Paper](https://arxiv.org/abs/2408.11227) | [💻 Code](https://github.com/ZucksLiu/OCTCubeM)  
_Modality:_ OCT

<details>
<summary><strong>Abstract</strong></summary>

We present OCTCube-M, a 3D OCT-based multi-modal foundation model for jointly analyzing OCT and en face images. OCTCube-M first developed OCTCube, a 3D foundation model pre-trained on 26,685 3D OCT volumes encompassing 1.62 million 2D OCT images. It then exploits a novel multi-modal contrastive learning framework COEP to integrate other retinal imaging modalities, such as fundus autofluorescence and infrared retinal imaging, into OCTCube, efficiently extending it into multi-modal foundation models. OCTCube achieves best performance on predicting 8 retinal diseases, demonstrating strong generalizability on cross-cohort, cross-device and cross-modality prediction. OCTCube can also predict cross-organ nodule malignancy (CT) and low cardiac ejection fraction as well as systemic diseases, such as diabetes and hypertension, revealing its wide applicability beyond retinal diseases. We further develop OCTCube-IR using COEP with 26,685 OCT and IR image pairs. OCTCube-IR can accurately retrieve between OCT and IR images, allowing joint analysis between 3D and 2D retinal imaging modalities. Finally, we trained a tri-modal foundation model OCTCube-EF from 4 million 2D OCT images and 400K en face retinal images. OCTCube-EF attains the best performance on predicting the growth rate of geographic atrophy (GA) across datasets collected from 6 multi-center global trials conducted in 23 countries. This improvement is statistically equivalent to running a clinical trial with more than double the size of the original study. Our analysis based on another retrospective case study reveals OCTCube-EF's ability to avoid false positive Phase-III results according to its accurate treatment effect estimation on the Phase-II results. In sum, OCTCube-M is a 3D multi-modal foundation model framework that integrates OCT and other retinal imaging modalities revealing substantial diagnostic and prognostic benefits.

</details>

---

**Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting (DINORET)**  
[📄 Paper](https://arxiv.org/abs/2409.17332) | [💻 Code](https://github.com/cm090999/dinoret)  
_Modality:_ CFP

<details>
<summary><strong>Abstract</strong></summary>

Integrating deep learning into medical imaging is poised to greatly advance diagnostic methods but it faces challenges with generalizability. Foundation models, based on self-supervised learning, address these issues and improve data efficiency. Natural domain foundation models show promise for medical imaging, but systematic research evaluating domain adaptation, especially using self-supervised learning and parameter-efficient fine-tuning, remains underexplored. Additionally, little research addresses the issue of catastrophic forgetting during fine-tuning of foundation models. We adapted the DINOv2 vision transformer for retinal imaging classification tasks using self-supervised learning and generated two novel foundation models termed DINORET and BE DINORET. Publicly available color fundus photographs were employed for model development and subsequent fine-tuning for diabetic retinopathy staging and glaucoma detection. We introduced block expansion as a novel domain adaptation strategy and assessed the models for catastrophic forgetting. Models were benchmarked to RETFound, a state-of-the-art foundation model in ophthalmology. DINORET and BE DINORET demonstrated competitive performance on retinal imaging tasks, with the block expanded model achieving the highest scores on most datasets. Block expansion successfully mitigated catastrophic forgetting. Our few-shot learning studies indicated that DINORET and BE DINORET outperform RETFound in terms of data-efficiency. This study highlights the potential of adapting natural domain vision models to retinal imaging using self-supervised learning and block expansion. BE DINORET offers robust performance without sacrificing previously acquired capabilities. Our findings suggest that these methods could enable healthcare institutions to develop tailored vision models for their patient populations, enhancing global healthcare inclusivity.

</details>

---

### 2023

**A foundation model for generalizable disease detection from retinal images (RETFound)**  
[📄 Paper](https://www.nature.com/articles/s41586-023-06555-x) | [💻 Code](https://github.com/rmaphoh/RETFound_MAE)  
_Modalities:_ CFP, OCT

<details>
<summary><strong>Abstract</strong></summary>

Medical artificial intelligence (AI) offers great potential for recognizing signs of health conditions in retinal images and expediting the diagnosis of eye diseases and systemic disorders. However, the development of AI models requires substantial annotation and models are usually task-specific with limited generalizability to different clinical applications. Here, we present RETFound, a foundation model for retinal images that learns generalizable representations from unlabelled retinal images and provides a basis for label-efficient model adaptation in several applications. Specifically, RETFound is trained on 1.6 million unlabelled retinal images by means of self-supervised learning and then adapted to disease detection tasks with explicit labels. We show that adapted RETFound consistently outperforms several comparison models in the diagnosis and prognosis of sight-threatening eye diseases, as well as incident prediction of complex systemic disorders such as heart failure and myocardial infarction with fewer labelled data. RETFound provides a generalizable solution to improve model performance and alleviate the annotation workload of experts to enable broad clinical AI applications from retinal imaging.

</details>

---

**VisionFM: a Multi-Modal Multi-Task Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence (VisionFM)**  
[📄 Paper](https://arxiv.org/abs/2310.04992) | [💻 Code](https://github.com/ABILab-CUHK/VisionFM)  
_Modalities:_ CFP, OCT, FFA, Slit-lamp, UBM, B-ultrasound, MRI, External eye

<details>
<summary><strong>Abstract</strong></summary>

We present VisionFM, a foundation model pre-trained with 3.4 million ophthalmic images from 560,457 individuals, covering a broad range of ophthalmic diseases, modalities, imaging devices, and demography. After pre-training, VisionFM provides a foundation to foster multiple ophthalmic artificial intelligence (AI) applications, such as disease screening and diagnosis, disease prognosis, subclassification of disease phenotype, and systemic biomarker and disease prediction, with each application enhanced with expert-level intelligence and accuracy. The generalist intelligence of VisionFM outperformed ophthalmologists with basic and intermediate levels in jointly diagnosing 12 common ophthalmic diseases. Evaluated on a new large-scale ophthalmic disease diagnosis benchmark database, as well as a new large-scale segmentation and detection benchmark database, VisionFM outperformed strong baseline deep neural networks. The ophthalmic image representations learned by VisionFM exhibited noteworthy explainability, and demonstrated strong generalizability to new ophthalmic modalities, disease spectrum, and imaging devices. As a foundation model, VisionFM has a large capacity to learn from diverse ophthalmic imaging data and disparate datasets. To be commensurate with this capacity, in addition to the real data used for pre-training, we also generated and leveraged synthetic ophthalmic imaging data. Experimental results revealed that synthetic data that passed visual Turing tests, can also enhance the representation learning capability of VisionFM, leading to substantial performance gains on downstream ophthalmic AI tasks. Beyond the ophthalmic AI applications developed, validated, and demonstrated in this work, substantial further applications can be achieved in an efficient and cost-effective manner using VisionFM as the foundation.

</details>

---

## License
- Linked code/models follow their original licenses.

