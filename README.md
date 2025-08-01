# Turkish Poetry LLM 

[![Made with Transformers](https://img.shields.io/badge/Made%20with-Transformers-ffbf00.svg)](https://huggingface.co/docs/transformers/index)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--08--01-blue.svg)](#changelog)

> A Turkish poetry language model fine-tuned with **LoRA** on top of GPT-2, optimized for **A100 (bf16)** with a clean developer-style pipeline and a Colab-ready notebook.

---

## 👤 Developer’s Note — **Adil Sevim**

**TR:**  
Bu projeye **13 yaşımda (2021)** bir istek üzerine başlamıştım. İlk sürümü birkaç gün içinde çıkardım. Elinizdeki sürüm, o dönemin mimarisini korurken **2025’te** kalite, güvenilirlik ve kullanım kolaylığı açısından kapsamlı bir gözden geçirme sürecinden geçti ve **01.08.2025** tarihinde paylaşıldı.  
**Sürümler:** 2021 (v1) → 2025 (revizyon & adaptasyon)

**EN:**  
I began this project at **age 13 (2021)** in response to a request and produced the first version within a few days. The current release preserves the original architecture and underwent a comprehensive **2025** review for quality, reliability, and usability. Published on **01 Aug 2025**.  
**Versions:** 2021 (v1) → 2025 (revision & adaptation)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Data](#-data)
- [Model & Training](#-model--training)
- [Environment & Installation](#-environment--installation)
- [Quickstart](#-quickstart)
- [Configuration Matrix](#-configuration-matrix)
- [Evaluation & Curves](#-evaluation--curves)
- [Generation Examples](#-generation-examples)
- [Repository Structure](#-repository-structure)
- [Exporting Weights](#-exporting-weights)
- [Responsible AI, Limitations](#-responsible-ai-limitations)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Changelog](#changelog)
- [Contact](#-contact)

---

## 🔎 Overview

**Turkish Poetry LLM** is a focused language model for **Turkish poetry generation**. It fine-tunes **GPT-2** with **LoRA adapters** and adds Turkish-specific special tokens to capture author styles, titles, and poem boundaries. The pipeline is **A100-optimized** (bf16) and **Colab-ready**, with frequent evaluation and best-model selection.

- **Started:** 2021 (first version at age 13)  
- **Public release:** 01 Aug 2025  
- **Use cases:** Poetry generation in the style of notable Turkish poets, prompt-conditioning by poet and title, experimentation with Turkish tokenization and structure.

---

## ✨ Features

- **LoRA fine-tuning** on GPT-2 (compact, memory-efficient, fast iterations)
- **A100-optimized precision:** `bf16` on A100, `fp16` elsewhere
- **Turkish special tokens:** poet, title, poem start/end (`<|şair:...|>`, `<|başlık|>`, `<|şiir|>`, `<|şiir_sonu|>`)
- **Frequent eval & best checkpoint** (cosine LR, warmup, weight decay)
- **Developer-style notebook:** clear sections, logs, and reproducibility hints

---

## 📚 Data

- **Primary dataset:** `Mrjavaci/Turkish-Poems` (JSON + TXT raw files)  
  - URL: `https://github.com/Mrjavaci/Turkish-Poems`  
- **Earlier iterations:** a dataset curated by **Adil Sevim** (private/legacy).
- **Preprocessing:** light cleaning, line joins, min-length filtering; mapping poet names to `<|şair:...|>` tokens; structure:  
  ```
  <|şair:Poet_Name|> <|başlık|> Title <|şiir|>
  poem content...
  <|şiir_sonu|>
  ```

> ℹ️ Please review the data source’s license and usage terms. Ensure you have the rights to use and redistribute any derived datasets or model outputs.

---

## 🧠 Model & Training

- **Base model:** `ytu-ce-cosmos/turkish-gpt2`  
- **LoRA target modules (GPT-2):** `["c_attn", "c_proj", "c_fc"]`  
- **Special tokens:** `"<|şair:...|>", "<|başlık|>", "<|şiir|>", "<|şiir_sonu|>", "<|mısra|>"`  
- **Max length:** `768` (memory-balanced)  
- **Loss:** Causal LM (no MLM)  
- **Scheduler:** cosine + warmup  
- **Precision:** `bf16` (A100), `fp16` otherwise

**Default hyperparameters (A100):**
- Epochs: `25`  
- Per-device batch: `4`  
- Grad accumulation: `16` (effective batch 64)  
- LR: `8e-5`  
- Warmup steps: `300`  
- Weight decay: `0.01`  
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`

**Non-A100 defaults:**
- Epochs: `15`  
- Per-device batch: `8`  
- Grad accumulation: `4`  
- LR: `1e-4`  
- Warmup steps: `150`  
- Precision: `fp16=True`  

---

## 🛠 Environment & Installation

```bash
pip install -U "transformers>=4.41.2" "accelerate>=0.31.0" "datasets>=2.20.0" \
  "peft>=0.12.0" "tokenizers>=0.19.1" torch matplotlib seaborn scikit-learn tqdm numpy pandas requests unidecode
```

> **Colab users:** After installing, **restart runtime** to ensure the new versions are active.

---

## ⚡ Quickstart

**Notebook (recommended):**  
`notebooks/Turkish_Poetry_LLM_Ultimate_A100_Developer.ipynb`

**Minimal training snippet:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
specials = ["<|başlık|>", "<|şiir|>", "<|şiir_sonu|>"]
tokenizer.add_special_tokens({'additional_special_tokens': specials})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.resize_token_embeddings(len(tokenizer))

lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
                      lora_dropout=0.05, target_modules=["c_attn","c_proj","c_fc"])
model = get_peft_model(model, lora_cfg)

# …prepare Dataset (input_ids, attention_mask, labels)…

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=25,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=8e-5,
    warmup_steps=300,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=30,
    save_strategy="steps",
    save_steps=60,
    save_total_limit=5,
    fp16=True,  # set bf16=True on A100
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,
)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=collator, tokenizer=tokenizer)
trainer.train()
```

**Generation:**
```python
def generate(prompt, max_new_tokens=200):
    import torch
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos_id = tokenizer.convert_tokens_to_ids("<|şiir_sonu|>")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.8,
                         top_p=0.9, top_k=50, repetition_penalty=1.1,
                         do_sample=True, pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=(eos_id if eos_id is not None else tokenizer.eos_token_id))
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    return text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)):]  # slice off the prompt
```

---

## 🧰 Configuration Matrix

| Setting     | A100 (Recommended) | Non-A100 |
|-------------|--------------------:|---------:|
| Precision   | bf16                | fp16     |
| Epochs      | 25                  | 15       |
| Per-dev BS  | 4                   | 8        |
| Accum Steps | 16                  | 4        |
| Effective BS| 64                  | 32       |
| LR          | 8e-5                | 1e-4     |
| Warmup      | 300                 | 150      |
| Max Length  | 768                 | 768      |

---

## 📊 Evaluation & Curves

- Primary metric: **validation loss** (reported every N steps)  
- Derived metric: **perplexity** = `exp(eval_loss)`  
- Plots: training/validation loss, validation perplexity, and LR schedule.

> Use `trainer.state.log_history` to reconstruct curves if needed.

---

## ✍️ Generation Examples

```text
<|şair:Nazim_Hikmet|> <|başlık|> Memleketimden İnsan Manzaraları <|şiir|>
…model output…
<|şiir_sonu|>

<|şair:Orhan_Veli|> <|başlık|> İstanbul'u Dinliyorum <|şiir|>
…model output…
<|şiir_sonu|>
```

> Tip: Keep temperature around `0.7–0.9` and `top_p≈0.9` for fluent, varied lines; add `<|başlık|>` for better topic focus.

---

## 📁 Repository Structure

```
.
├── notebooks/
│   └── Turkish_Poetry_LLM.ipynb
├── src/                      # (optional) helpers, training scripts
├── data/                     # raw/processed data (gitignored)
├── models/                   # checkpoints (use Git LFS or HF Hub)
├── README.md
├── LICENSE
└── requirements.txt
```

> Large artifacts should go to **Hugging Face Hub** or use **Git LFS**.

---

## 📦 Exporting Weights

- Save with `trainer.save_model()` and `tokenizer.save_pretrained(...)`.
- Optionally pack as `.zip` for distribution.
- Respect base model licensing for redistribution of weights.

---

## 🤝 Responsible AI, Limitations

- **Style imitation:** The model can produce text reminiscent of well-known poets; generated text may unintentionally resemble copyrighted works.  
- **Bias & content risk:** Trained on human-authored poems; may inherit cultural biases or produce unexpected content.  
- **Use responsibly:** Review outputs before publication. Do not imply endorsement by real authors.

---

### Model Weights

You can download the trained AI model from Google Drive: [Download the model](https://drive.google.com/file/d/1i6OVXh1vXFOf1CqeuhiS7n6zjH91PPgq/view?usp=drive_link).

---

## 👏 Contributing

PR’ler ve issue’lar hoş geldiniz. Lütfen açıklayıcı başlıklar ve kısa örnekler ekleyin. Büyük model dosyalarını yüklemeyin; HF Hub kullanın.

---

## 📄 License

- **Code:** MIT (recommended) — see [LICENSE](./LICENSE)  
- **Model Weights & Data:** Respect the base model and dataset licenses.  
  - Base: `ytu-ce-cosmos/turkish-gpt2` (check its license)  
  - Data: `Mrjavaci/Turkish-Poems` (review repository terms)

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@software{adil2021turkishpoetryllm,
  author  = {Adil Sevim},
  title   = {Turkish Poetry LLM },
  year    = {2025},
  month   = {August},
  url     = {https://github.com/AdilSevim/Turkish-Poetry-Language-Model}
}
```

---

## 🙏 Acknowledgements

- **Mrjavaci** for the Turkish poems dataset.  
- Hugging Face ecosystem and open-source contributors.

---

## Changelog

- **2025-08-01:** Public release with developer-style notebook and LoRA pipeline.

---

## 📬 Contact

**Adil Sevim** — Developer  
- GitHub: https://github.com/<AdilSevim>  
- LinkedIn: https://www.linkedin.com/in/adilsevim/  
- Email: <adilsevim18@gmail.com>
