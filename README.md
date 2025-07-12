# Linkedin-post-generator-LLM

A fine-tuned Large Language Model (DistilGPT2) for generating professional, human-like LinkedIn posts without external API dependencies.

## Project Overview

This project focuses on designing and developing a Large Language Model (LLM) capable of generating human-like, professional LinkedIn posts across various themes. The core objective of this assignment was to demonstrate the ability to build and deploy an LLM that operates entirely without dependency on external APIs (e.g., OpenAI, Anthropic) for its generative capabilities, ensuring full local control over the model.

## Approach

Given the scope of a solo project, a **transfer learning** approach was adopted. Instead of training an LLM from scratch (which requires immense computational resources and time), a smaller, open-source, pre-trained language model (`DistilGPT2`) was fine-tuned on a custom-curated dataset. This method efficiently adapts a powerful base model to a highly specific task and style (professional LinkedIn posts).

## Technical Stack

* **Python:** Primary programming language.
* **Hugging Face Transformers:** Used for loading pre-trained models, tokenization, and managing the fine-tuning process.
* **Hugging Face `datasets`:** For efficient handling and processing of the custom dataset.
* **PyTorch:** The underlying deep learning framework used by the model.
* **Gradio:** For creating a simple, interactive web interface to demonstrate the model's functionality.

## Dataset

A custom high-quality dataset for professional LinkedIn posts was meticulously curated. The dataset was structured in a `JSONL` format, with each entry containing a `prompt` (the topic or instruction for the post) and a `completion` (the desired LinkedIn post). This facilitated instruction fine-tuning, allowing the model to learn to generate specific types of posts based on an input prompt.

**Themes Covered Include:**
* Career Growth & Learning
* Industry Insights & Trends
* Team Achievements & Collaboration
* Personal Branding & Networking
* Hiring Opportunities

The dataset can be found at `linkedin_posts.jsonl` in this repository.

## Model Details

* **Base Model:** `DistilGPT2` (an 82M parameter model, a distilled version of GPT-2).
* **Fine-tuning:** The `DistilGPT2` model was fine-tuned for `[Number, e.g., 5]` epochs on the custom LinkedIn post dataset.

## Dataset Example

{"prompt": "Write a LinkedIn post about networking.", "completion": "Networking success! 🌐 Attended an incredible conference last week and met some amazing professionals. Connected with so many talented individuals who share my passion for innovation. 🤝 #Networking #Collaboration"}

## Limitations

Despite successfully demonstrating a functional LinkedIn post generator, this project operates under several key limitations, primarily stemming from resource constraints and the inherent nature of fine-tuning smaller models. Foremost among these is the dataset size and composition, with only 190 custom entries. While meticulously curated for quality, this limited volume restricts the model's exposure to diverse linguistic nuances, professional contexts, and a wider range of tone variations prevalent on LinkedIn. The custom nature of the entries, while ensuring relevance, means the model's output might be heavily biased towards the specific writing style and themes present in this small dataset, potentially leading to a lack of genuine creativity or the inability to handle prompts significantly outside its learned patterns. Furthermore, the use of DistilGPT2, an 82-million parameter model, inherently limits its capacity to grasp complex long-range dependencies, intricate reasoning, or subtle contextual understanding compared to much larger LLMs. This can sometimes result in generic, repetitive, or logically inconsistent outputs, especially with less common prompts or if specific generation parameters are not optimally tuned. Finally, while the fine-tuning process helps adapt the model, it does not imbue it with real-world knowledge beyond what's implicitly captured in the base model and explicitly in the small fine-tuning dataset, meaning it lacks up-to-date factual awareness or deep domain expertise for highly specialized post generation.
