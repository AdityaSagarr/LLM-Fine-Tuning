

# 🚀 **LLM Fine-Tuning with LoRA and qLoRA: A Complete Guide**

Fine-tuning **Large Language Models (LLMs)** like **Llama-2-7b** can be memory-intensive and computationally expensive. **LoRA (Low-Rank Adaptation)** and **qLoRA (Quantized LoRA)** provide efficient methods to adapt pre-trained models without overburdening system resources. This guide will walk you through the process of fine-tuning using these techniques in **Google Colab**.

---

## 📘 **What is LoRA and qLoRA?**

### **LoRA (Low-Rank Adaptation)**  
LoRA is a technique that reduces the number of parameters updated during fine-tuning by introducing **low-rank matrices** into the model. This results in efficient use of memory while retaining the ability to adapt the model for a new task.

- **Why use LoRA?**  
  LoRA helps save computational resources while still enabling meaningful fine-tuning of large models.

### **qLoRA (Quantized LoRA)**  
qLoRA extends LoRA by using **4-bit quantization**, which further reduces memory usage. This is especially useful when fine-tuning extremely large models, allowing them to fit into systems with limited GPU memory.

- **Why use qLoRA?**  
  With **4-bit quantization**, qLoRA significantly reduces memory requirements and computation time, making it possible to fine-tune larger models on less powerful hardware.

---

## 📦 **1. Install Required Packages**

Before starting, install the necessary libraries for fine-tuning, including **LoRA**, **qLoRA**, and **TensorBoard** for visualization.

```bash
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 tensorboard
```

### Libraries Used:
- **accelerate**: Simplifies model training on different hardware setups.
- **peft**: Implements LoRA (Low-Rank Adaptation).
- **bitsandbytes**: Provides quantization capabilities for qLoRA.
- **transformers**: Hugging Face's library for pre-trained models.
- **trl**: Used for Reinforcement Learning (RL) fine-tuning.
- **tensorboard**: To visualize training progress.

---

## 🧑‍💻 **2. Setup and Load Dataset**

Next, we load the dataset, split it into training and validation sets, and prepare it for fine-tuning.

```python
from datasets import load_dataset

# Load the dataset
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.train_test_split(test_size=0.1)  # 10% for evaluation

# Split dataset
train_dataset = dataset['train']
eval_dataset = dataset['test']
```

### **Dataset Explanation:**
- **Training Set**: 90% of the dataset for fine-tuning the model.
- **Evaluation Set**: 10% of the dataset for monitoring model performance during training.

---

## ⚙️ **3. LoRA and qLoRA Configuration**

Now, we configure the LoRA and qLoRA settings for efficient training.

### **LoRA Parameters**:
```python
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,  # Controls how much the low-rank matrices affect the model
    lora_dropout=0.1,  # Dropout to prevent overfitting
    r=64,  # Rank of the low-rank matrix, determines how much the model is adjusted
    bias="none",  # No bias added to the low-rank matrices
    task_type="CAUSAL_LM"  # Task type for causal language modeling (text generation)
)
```

### **qLoRA Parameters**:
```python
from bitsandbytes import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enables 4-bit precision quantization
    bnb_4bit_quant_type="nf4",  # Defines the quantization type (nf4)
    bnb_4bit_compute_dtype=torch.float16,  # 16-bit computation precision
    bnb_4bit_use_double_quant=False  # Disables double quantization for simpler optimization
)
```

### **Explanation of Key Parameters**:
- **`lora_alpha` (16)**: Scaling factor to control the influence of the low-rank matrices.
- **`lora_dropout` (0.1)**: Dropout to prevent overfitting and improve generalization.
- **`r` (64)**: Rank of the low-rank matrices that defines the degree of adaptation.
- **`bnb_4bit_compute_dtype` (float16)**: Precision used during computations for faster training.

---

## 🔥 **4. Load Pre-trained Model and Tokenizer**

We load the model and tokenizer from the Hugging Face model hub, and apply the configurations we just defined.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with quantization and LoRA
model_name = "NousResearch/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}  # Use the first available GPU
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS
tokenizer.padding_side = "right"  # Padding side for sequences
```

### **Tokenizer Setup**:
- **Pad Token**: The `pad_token` is set to the `eos_token` (end-of-sequence) for consistency in tokenization.
- **Padding Side**: Sequences will be padded on the right side.

---

## 🎯 **5. Training Arguments Setup**

Configure the parameters that control the fine-tuning process, such as batch size, learning rate, and evaluation steps.

```python
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="./results",  # Save model and logs
    num_train_epochs=1,  # 1 epoch of training (1 full pass through the dataset)
    per_device_train_batch_size=4,  # Batch size per device (for training)
    per_device_eval_batch_size=4,  # Batch size per device (for evaluation)
    gradient_accumulation_steps=1,  # Steps to accumulate gradients before updating the model
    optim="paged_adamw_32bit",  # Optimizer for training
    logging_steps=25,  # Log every 25 steps
    save_steps=0,  # Do not save checkpoints at each step
    learning_rate=2e-4,  # Learning rate controls how fast the model learns
    weight_decay=0.001,  # Regularization to prevent overfitting
    max_grad_norm=0.3,  # Gradient clipping to avoid exploding gradients
    warmup_ratio=0.03,  # 3% of training for learning rate warmup
    lr_scheduler_type="cosine",  # Cosine learning rate decay
    report_to="tensorboard",  # Report to TensorBoard for visualization
    evaluation_strategy="steps",  # Evaluate at intervals
    eval_steps=25,  # Evaluate every 25 steps
    do_eval=True  # Enable evaluation
)
```

### **Key Parameters**:
- **`per_device_train_batch_size` (4)**: Batch size during training. A batch size of 4 helps manage memory usage.
- **`learning_rate` (2e-4)**: Controls how fast the model updates its parameters.
- **`weight_decay` (0.001)**: Helps prevent overfitting by penalizing large weights.

---

## ⚡ **6. Fine-Tuning the Model**

Set up the trainer and start the fine-tuning process. After training, we save the fine-tuned model.

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Use the evaluation set for monitoring
    peft_config=peft_config,
    dataset_text_field="text",  # Specify the text field in the dataset
    tokenizer=tokenizer,
    args=training_arguments
)

trainer.train()  # Start training
trainer.model.save_pretrained("Llama-2-7b-chat-finetune")  # Save the fine-tuned model
```

---

## 📊 **7. Visualize Training with TensorBoard**

Use TensorBoard to visualize the training process in real time.

```python
%load_ext tensorboard  # Load TensorBoard extension

%tensorboard --logdir ./results  # Open TensorBoard to see the training logs
```

---

## 🎉 **Conclusion**

Using **LoRA** and **qLoRA**, we can fine-tune large language models like **Llama-2-7b** efficiently. By leveraging **low-rank adaptation** and **4-bit quantization**, we reduce memory usage while maintaining model performance. With this guide, you can quickly adapt models to new tasks using **Google Colab** or any other setup with limited GPU resources.

Happy fine-tuning!
