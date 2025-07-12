import os
import json
import torch
import gradio as gr
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer 


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "linkedin_posts.jsonl")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "fine_tuned_model")
BASE_MODEL_NAME = "distilgpt2"
MAX_SEQ_LENGTH = 256 

TRAINING_ARGS = TrainingArguments(
    output_dir=os.path.join(PROJECT_DIR, "results"), 
    num_train_epochs=5,           
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=8,
    learning_rate=5e-5,           
    warmup_steps=100,             
    weight_decay=0.01,            
    logging_dir=os.path.join(PROJECT_DIR, "logs"), 
    logging_steps=50,             
    save_strategy="epoch",       
    load_best_model_at_end=True,  
    metric_for_best_model="loss", 
    greater_is_better=False,      
    fp16=True,                   
    report_to="none"            
)

#data preparation
def load_and_prepare_data(file_path):
    print(f"Loading data from {file_path}...")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} entries.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please ensure it exists.")
        exit() 

    
    dataset = Dataset.from_list(data)

    
    def format_example(example):
        return {"text": f"Prompt: {example['prompt']}\nCompletion: {example['completion']}{tokenizer.eos_token}"}

   
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_dataset = dataset.map(format_example).map(
        tokenize_function, batched=True, remove_columns=["prompt", "completion"]
    )
    print("Data prepared and tokenized.")
    return tokenized_dataset, tokenizer

#model fine tuning
def fine_tune_model(train_dataset, tokenizer):
    print(f"Loading base model: {BASE_MODEL_NAME} for fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Starting model training...")
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("Fine-tuning complete!")

   

    print(f"Saving fine-tuned model to {MODEL_OUTPUT_DIR}...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR) 
    print("Model saved.")
    return model

#local
def load_fine_tuned_model_for_inference():
    print(f"Loading fine-tuned model and tokenizer from: {MODEL_OUTPUT_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_OUTPUT_DIR)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print(f"Model loaded successfully to device: {device}")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model from {MODEL_OUTPUT_DIR}: {e}")
        print("Please ensure the model has been fine-tuned and saved correctly in this directory.")
        print("If you haven't fine-tuned yet, set TRAIN_MODEL = True in main().")
        exit()

def generate_linkedin_post(prompt_text, model, tokenizer, device, max_length=200, temperature=0.7, top_k=50, top_p=0.95):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    formatted_prompt = f"Prompt: {prompt_text}\nCompletion:"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    generated_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id 
    )

    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    completion_start_tag = "\nCompletion:"
    if completion_start_tag in generated_text:
        post = generated_text.split(completion_start_tag, 1)[1].strip()
    else:
        post = generated_text.strip()
    if "Prompt:" in post:
        post = post.split("Prompt:", 1)[0].strip()

    return post

#gradio
def launch_gradio_interface(model, tokenizer, device):
    print("Launching Gradio interface...")
    def predict_post(prompt_input, max_len, temp, top_k_val, top_p_val):
        return generate_linkedin_post(
            prompt_text=prompt_input,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=int(max_len),
            temperature=float(temp),
            top_k=int(top_k_val),
            top_p=float(top_p_val)
        )

    iface = gr.Interface(
        fn=predict_post,
        inputs=[
            gr.Textbox(lines=3, placeholder="E.g., 'Write a post about effective leadership in remote teams.'", label="Prompt for LinkedIn Post"),
            gr.Slider(minimum=50, maximum=300, value=200, label="Max Post Length"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="Temperature (Creativity)"),
            gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K Sampling"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-P Sampling")
        ],
        outputs="text",
        title="LinkedIn Post Generator (Fine-tuned DistilGPT2)",
        description="Enter a prompt to generate a professional, human-like LinkedIn post. This model runs locally after fine-tuning."
    )

    iface.launch(share=False)
    print("Gradio interface launched. Check your browser for the local URL.")



def main():
    TRAIN_MODEL = True
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please create it or place it in the '{DATA_DIR}' folder.")
        print("Expected format: Each line is a JSON object like {'prompt': '...', 'completion': '...'}")
        print("Example: echo '{\"prompt\": \"career growth\", \"completion\": \"Continuous learning is key... #Growth\"}' > data/linkedin_posts.jsonl")
        return

    tokenized_dataset, tokenizer_for_training = load_and_prepare_data(DATA_FILE)

    model = None
    if TRAIN_MODEL:
        model = fine_tune_model(tokenized_dataset["train"], tokenizer_for_training)
    else:
        print("Skipping training. Attempting to load existing fine-tuned model...")
        model, tokenizer_for_inference, device = load_fine_tuned_model_for_inference()
        tokenizer_for_training = tokenizer_for_inference
    
    if TRAIN_MODEL or model is None: 
        model, tokenizer_for_inference, device = load_fine_tuned_model_for_inference()
        tokenizer_for_training = tokenizer_for_inference 

    if model and tokenizer_for_training:
        launch_gradio_interface(model, tokenizer_for_training, device)
    else:
        print("Model or tokenizer not available for Gradio interface. Please check logs.")

if __name__ == "__main__":
    main()