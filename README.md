# distilbert-base-uncased-finetuned-emotions-dataset

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) trained on the **Emotion** dataset to classify text inputs into emotional categories.

---

## Model Performance

| Metric   | Value  |
|----------|--------|
| Loss     | 0.2428 |
| Accuracy | 0.9395 |
| F1 Score | 0.9396 |

---

## Model Description

The model is fine-tuned to perform **text classification** specifically for emotional content. It categorizes input text into one of six emotional classes with high accuracy and F1 score.

---

## Intended Uses & Limitations

### Intended Uses
- Sentiment and emotion analysis in text
- Emotion-aware chatbots or recommendation systems
- Social media analysis and monitoring emotional trends

### Limitations
- The model may inherit biases from the training data.
- It is optimized for six basic emotions and may not capture complex or subtle emotional nuances.
- Performance may degrade on out-of-domain or very informal text.

---

## Training and Evaluation Data

- **Dataset:** [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)  
- **Emotional Categories:**
  - `LABEL_0`: sadness
  - `LABEL_1`: joy
  - `LABEL_2`: love
  - `LABEL_3`: anger
  - `LABEL_4`: fear
  - `LABEL_5`: surprise

### Dataset Description
The dataset contains English Twitter messages labeled with six basic emotions (anger, disgust, fear, joy, sadness, surprise). The goal is to classify tweets into these emotional categories.

---

## Training Details

### Hyperparameters
- Learning rate: 2e-5
- Train batch size: 32
- Evaluation batch size: 32
- Optimizer: Adam (betas=(0.9, 0.999), epsilon=1e-8)
- Learning rate scheduler: Linear
- Number of epochs: 10
- Random seed: 42

### Training Progress

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|-------|---------------|-----------------|----------|----------|
| 1     | 0.5929        | 0.2345          | 0.9185   | 0.9180   |
| 2     | 0.1642        | 0.1716          | 0.9335   | 0.9342   |
| 3     | 0.1163        | 0.1501          | 0.9405   | 0.9407   |
| 4     | 0.0911        | 0.1698          | 0.9330   | 0.9331   |
| 5     | 0.0741        | 0.1926          | 0.9320   | 0.9323   |
| 6     | 0.0559        | 0.2033          | 0.9350   | 0.9353   |
| 7     | 0.0464        | 0.2156          | 0.9350   | 0.9353   |
| 8     | 0.0335        | 0.2354          | 0.9405   | 0.9408   |
| 9     | 0.0257        | 0.2410          | 0.9395   | 0.9396   |
| 10    | 0.0214        | 0.2428          | 0.9395   | 0.9396   |

---

## Examples

Try these example texts to see emotion predictions:

| Example Title | Text                                                        |
|---------------|-------------------------------------------------------------|
| Example 1     | "on a boat trip to denmark"                                 |
| Example 2     | "i was feeling listless from the need of new things something different" |
| Example 3     | "i know im feeling agitated as it is from a side effect of the too high dose" |

---

## Requirements

- transformers==4.35.2
- torch==2.1.0+cu118
- datasets==2.15.0
- tokenizers==0.15.0

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Contact

For questions or feedback, please open an issue or contact the maintainer.

---

<!-- This README was generated based on the training and model metadata. Please review and customize further if needed. -->
