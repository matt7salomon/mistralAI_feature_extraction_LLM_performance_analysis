# mistralAI_feature_extraction_LLM_performance_analysis

The dataset can be downloaded here and I used the original data not the preprocessed one:
https://github.com/ruidan/Unsupervised-Aspect-Extraction/tree/master


## Project Overview

This project is designed to leverage **Mistral** (a large language model) and various tools from **LangChain**, **HuggingFace**, and **Pandas** to accomplish two key tasks:
1. **Information Extraction**: Extracting structured data from unstructured medical notes and restaurant reviews.
2. **Evaluation**: Comparing model responses to predefined golden answers and calculating accuracy rates across test cases.

It uses **Selenium** to scrape data, **Mistral** for language model inference, and various libraries for manipulating and analyzing the extracted data.

## Theoretical Explanation of Key Functions and Calls

### 1. **Environment Setup**
```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
```
- **Purpose**: Loads environment variables from a `.env` file, specifically the Mistral API key used to interact with the Mistral API.
- **Theoretical Context**: Loading environment variables this way allows you to avoid hardcoding sensitive information like API keys into the codebase, improving security and flexibility.

### 2. **Data Initialization with Prompts**
```python
prompts = {
    "Johnson": {
        "medical_notes": "A 60-year-old male patient, Mr. Johnson...",
        "golden_answer": {
            "age": 60, "gender": "male", "diagnosis": "diabetes", "weight": 210, "smoking": "yes"
        }
    },
    "Smith": { ... }
}
```
- **Purpose**: A dictionary containing medical notes and expected ("golden") answers in structured JSON format. Each entry simulates a medical case.
- **Theoretical Context**: This setup acts as ground truth data for evaluating model performance. The golden answers represent ideal responses, against which the model-generated responses will be compared.

### 3. **Mistral Model API Call**
```python
def run_mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat.complete(
        model=model, messages=messages, response_format={"type": "json_object"}
    )
    return chat_response.choices[0].message.content
```
- **Purpose**: Sends a user message (prompt) to the Mistral model using the API, then retrieves the model’s response in JSON format.
- **Theoretical Context**: Large language models like Mistral are capable of performing tasks such as information extraction, summarization, and structured data generation. By setting the response format as `json_object`, we force the model to return a structured, machine-readable output.

### 4. **Prompt Template Definition**
```python
prompt_template = """
Extract information from the following medical notes:
{medical_notes}
Return json format with the following JSON schema: 
{ ... }
"""
```
- **Purpose**: Defines a template for the prompts sent to the Mistral model, including the structure and format of the expected output.
- **Theoretical Context**: By constraining the model’s response to a well-defined JSON schema, we increase the likelihood that the model produces consistent, structured outputs. This is important for downstream tasks like accuracy comparison.

### 5. **Response Comparison and Evaluation**
```python
def compare_json_objects(obj1, obj2):
    total_fields = 0
    identical_fields = 0
    common_keys = set(obj1.keys()) & set(obj2.keys())
    for key in common_keys:
        identical_fields += obj1[key] == obj2[key]
    percentage_identical = (identical_fields / max(len(obj1.keys()), 1)) * 100
    return percentage_identical
```
- **Purpose**: Compares the JSON output from the model to the golden answers, calculating the percentage of matching fields between them.
- **Theoretical Context**: This function performs field-wise comparison to determine how closely the model’s output aligns with the predefined correct answer (golden answer). The comparison focuses on key-value pairs and computes a similarity score based on how many fields match.

### 6. **Main Evaluation Loop**
```python
accuracy_rates = []
for name in prompts:
    user_message = prompt_template.format(medical_notes=prompts[name]["medical_notes"])
    response = json.loads(run_mistral(user_message))
    accuracy_rates.append(compare_json_objects(response, prompts[name]["golden_answer"]))
sum(accuracy_rates) / len(accuracy_rates)
```
- **Purpose**: Iterates over the predefined medical cases (`prompts`), runs each case through the Mistral model, and compares the model’s response to the golden answer, computing an overall accuracy score.
- **Theoretical Context**: This process allows for the evaluation of model performance by using test cases and comparing results. The average accuracy rate is calculated to determine the model’s overall effectiveness in extracting the correct information from the medical notes.

### 7. **Another Use Case: Restaurant Review Dataset**
```python
df_label = load_dataset(input_dir/'test_label.txt')
df = load_dataset(input_dir/'test.txt')
df = pd.concat([df, df_label], axis=1)
```
- **Purpose**: Loads and prepares a dataset of restaurant reviews and their corresponding labels. The labels include aspects like “Ambience,” “Food,” and “Staff.”
- **Theoretical Context**: This section shows a broader use case of text analysis where aspects (e.g., ambience, food) are extracted from text. In NLP, aspect extraction is critical for sentiment analysis, recommendation systems, and business analytics.

### 8. **Language Model Setup: HuggingFace Pipeline**
```python
from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(pipeline=pipe)
```
- **Purpose**: Sets up a text generation pipeline using HuggingFace’s transformers library. The pipeline is initialized for text generation tasks.
- **Theoretical Context**: HuggingFace’s transformer models provide state-of-the-art performance in various NLP tasks. By using a pipeline for text generation, this code creates a flexible model capable of generating text based on user input or extracted document content.

### 9. **Evaluation of Aspect Extraction**
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
for label in interested_labels:
    y_true = df_final[label]
    y_pred = df_final[f'predicted_{label}']
    print('Accuracy:', accuracy_score(y_true, y_pred))
    ...
```
- **Purpose**: Evaluates the performance of aspect extraction using classification metrics such as **accuracy**, **F1 score**, **precision**, and **recall**.
- **Theoretical Context**: These metrics are essential for evaluating any classification task. **Accuracy** measures the overall correctness, **precision** measures how many of the predicted labels were correct, **recall** measures how many of the true labels were predicted, and **F1 score** is the harmonic mean of precision and recall, providing a balanced measure.

## Conclusion
This project demonstrates the use of advanced NLP techniques for structured information extraction from unstructured text (medical notes and restaurant reviews). It leverages **Mistral** for language understanding and extraction, **LangChain** for pipeline orchestration, and **Pandas** for dataset manipulation. The accuracy and effectiveness of the system are evaluated using ground truth data (golden answers) and traditional classification metrics, making it a robust pipeline for similar NLP tasks.

## Next Steps
1. **Model Fine-Tuning**: Consider fine-tuning the Mistral model for better domain-specific accuracy.
2. **Expand Dataset**: Increase the size and diversity of test cases to enhance model evaluation.
3. **Implement Real-Time Application**: Integrate the pipeline into a web service or API for real-time use in medical or business applications.

