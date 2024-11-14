# Product Review Classification using Transfer Learning with ELMo, ULMFiT, and BERT

## Overview
This study explores the use of transfer learning models—**ELMo**, **ULMFiT**, and **BERT**—for the classification of product reviews. Each model brings distinct capabilities for handling language nuances, with **FastText** integrated with ULMFiT as a baseline. The research evaluates how these models handle the complexities of product reviews, such as sentiment, sarcasm, and informal language, to improve classification accuracy.

## Table of Contents
1. [Objective](#objective)
2. [Models and Methodology](#models-and-methodology)
3. [Dataset and Preprocessing](#dataset-and-preprocessing)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [References](#references)

## Objective
The primary aim is to address the limitations of traditional text classification methods by leveraging **ELMo**, **ULMFiT**, and **BERT** models, each designed to capture contextual information in unique ways. This study compares their effectiveness in capturing the subtleties of language in product reviews and enhancing classification accuracy.

## Models and Methodology

### 1. ELMo
ELMo (Embeddings from Language Models) generates contextual embeddings using bidirectional LSTMs to capture word meaning within context. The embeddings are fine-tuned using a simple neural network for binary sentiment classification (positive or negative).

- **Training**: Fine-tuned using binary cross-entropy loss with the Adam optimizer.
- **Accuracy**: Achieved 90% accuracy on the validation set.

### 2. ULMFiT
ULMFiT (Universal Language Model Fine-tuning) applies a two-stage fine-tuning process for transfer learning:
   - **Language Model Fine-Tuning**: The model adapts to the Amazon reviews domain using a high initial learning rate and a cosine annealing schedule.
   - **Classification Fine-Tuning**: A classification head is trained with discriminative fine-tuning and gradual unfreezing for optimal domain adaptation.

- **Optimizer**: SGD with momentum during language model fine-tuning; Adam for classification fine-tuning.
- **Accuracy**: Achieved the highest accuracy of 94% with FastText embeddings.

### 3. BERT
BERT (Bidirectional Encoder Representations from Transformers) utilizes self-attention mechanisms to capture bidirectional context, making it well-suited for complex language tasks like sentiment analysis.

- **Training**: Fine-tuned with the AdamW optimizer, using warm-up and decay learning rate schedules.
- **Accuracy**: Reached 92% accuracy on the validation set.

## Dataset and Preprocessing
The dataset consists of Amazon product reviews labeled by sentiment:
   - **Positive**: 4- or 5-star ratings
   - **Negative**: 1- or 2-star ratings
   - **Neutral (3-star)**: Excluded to avoid ambiguity

The dataset is split into training and validation sets to evaluate model performance.

## Results
| Model         | Accuracy |
|---------------|----------|
| ELMo          | 90%      |
| BERT          | 92%      |
| ULMFiT + FastText | 94%  |

**ULMFiT** with **FastText embeddings** achieved the highest accuracy (94%), suggesting that its two-stage fine-tuning approach aligns well with domain-specific nuances.

## Conclusion
This study concludes that while BERT provides strong general language comprehension, **ULMFiT** with domain-specific fine-tuning is more effective for product review classification. **ELMo** serves as a computationally efficient alternative but lags in accuracy. This work emphasizes the importance of selecting models based on both accuracy and computational demands, as well as the potential of combined transfer learning strategies.

## References
1. Alshattnawi, S., et al. (2024). Beyond Word-Based Model Embeddings: Contextualized Representations for Enhanced Social Media Spam Detection. *Applied Sciences, 14*(6), 2254.
2. de Barros, T. M., et al. (2021). Leveraging emoji to improve sentiment classification of tweets. *Proceedings of the 36th annual ACM symposium on applied computing.*
3. Othan, D., & Kilimci, Z. H. (2021). Stock market prediction with deep contextualized word representations. *INISTA, IEEE.*
4. Malhotra, S., et al. (2021). Bidirectional transfer learning model for sentiment analysis. *Journal of Ambient Intelligence and Humanized Computing, 12*(11), 10267-10287.

(*Further references continue as needed*)
