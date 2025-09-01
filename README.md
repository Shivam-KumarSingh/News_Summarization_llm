# Abstractive News Summarization using BART

This project contains the code to fine-tune a BART (Bidirectional and Auto-Regressive Transformer) model for abstractive news summarization. The model is trained on the ILSUM dataset using the HuggingFace `transformers` and `datasets` libraries.



[Image of a neural network processing text]


## Description

The goal of this project is to create a model that can generate concise, coherent summaries of long-form news articles. Unlike extractive summarization, which pulls sentences directly from the source, this abstractive model generates new sentences that capture the core meaning of the text. This is achieved by fine-tuning a pre-trained `facebook/bart-base` model on a domain-specific dataset.

## Features

-   **End-to-End Pipeline:** Complete workflow from data loading and preprocessing to model training, evaluation, and inference.
-   **HuggingFace Integration:** Leverages the powerful `transformers`, `datasets`, and `evaluate` libraries for an industry-standard workflow.
-   **Performance Metrics:** Uses ROUGE scores for a standardized and rigorous evaluation of the model's summarization quality.
-   **Optimized Training:** Configured with `Seq2SeqTrainingArguments` for efficient training, including support for mixed-precision (`fp16`) on GPUs.

## Tech Stack

-   **Python**
-   **PyTorch**
-   **HuggingFace Transformers**
-   **HuggingFace Datasets**
-   **HuggingFace Evaluate**
-   **NumPy**

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Shivam-KumarSingh/News_Summarization_llm.git
    cd News_Summarization_llm
    ```

2.  **Install the required packages:**
    ```bash
    pip install transformers datasets evaluate rouge_score torch
    ```
    *(Note: Ensure you have a compatible version of PyTorch installed for your hardware, especially if using a GPU).*

## Usage

The entire pipeline is contained within the  Jupyter Notebook.

1.  **Run the script:** Execute the cells in the notebook .
    

2.  **Training Process:** The Notebook cell will automatically:
    -   Load the `ILSUM/ILSUM-1.0` dataset.
    -   Filter and preprocess the articles and summaries.
    -   Tokenize the text using the BART tokenizer.
    -   Initialize the `Seq2SeqTrainer`.
    -   Start the fine-tuning process for 3 epochs(adjust parameters and epochs for training to get better output ).
    -   Save the trained model and tokenizer to the output directory (`./bart-news-summarizer`).

3.  **Inference:** The last part of the script demonstrates how to load the fine-tuned model and generate a summary for a sample article.

## Results

The model was evaluated on the test set, achieving the following ROUGE scores, which measure the overlap between the generated and reference summaries.

| Metric      | Score      |
| :---------- | :--------- |
| **ROUGE-1** | **37.26** |
| **ROUGE-2** | **27.50** |
| **ROUGE-L** | **34.40** |

These results demonstrate the model's strong capability in generating relevant and accurate abstractive summaries.
