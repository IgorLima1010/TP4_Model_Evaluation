import pytest
import deepeval
from deepeval.test_case import LLMTestCase
from unsloth import FastLanguageModel
from peft import PeftModel
import json
import torch

from execution_metric import ExecutionAccuracyMetric

# --- CONFIGURAÇÃO ---
NOME_DO_MODELO_BASE = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
PATH_ADAPTADOR_1 = "lora_model_version1"
PATH_ADAPTADOR_2 = "lora_model_version2"
PROMPT_TEMPLATE_FILE = "prompt_template.txt"

with open(PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
    prompt_template = f.read()
with open('dev.json', 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

def load_lora_model(base_model_name, adapter_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model_1, tokenizer_1 = load_lora_model(NOME_DO_MODELO_BASE, PATH_ADAPTADOR_1)
model_2, _ = load_lora_model(NOME_DO_MODELO_BASE, PATH_ADAPTADOR_2)

modelos_para_testar = [
    ("Modelo 1 (lora_model_version1)", model_1, tokenizer_1),
    ("Modelo 2 (lora_model_version2)", model_2, tokenizer_1),
]

def gerar_sql(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=256)
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    try:
        assistant_part = full_response.split("assistant\n\n")[-1].strip()
        if assistant_part.lower().startswith("```sql"):
            assistant_part = assistant_part[len("```sql"):].strip()
        if assistant_part.endswith("```"):
            assistant_part = assistant_part[:-3].strip()
        if assistant_part.endswith(';'):
            assistant_part = assistant_part[:-1]
        return assistant_part
    except Exception:
        return full_response

@pytest.mark.parametrize("item_teste", dev_data)
@pytest.mark.parametrize("nome_modelo, modelo, tokenizer", modelos_para_testar)
def test_spider_execution(nome_modelo, modelo, tokenizer, item_teste):
    pergunta = item_teste['question']
    sql_correta = item_teste['query']
    db_id = item_teste['db_id']
    
    prompt_final = prompt_template.replace('{your_new_question_here}', pergunta)
    sql_gerada = gerar_sql(modelo, tokenizer, prompt_final)

    caso_de_teste = LLMTestCase(
        input=pergunta,
        actual_output=sql_gerada,
        expected_output=sql_correta,
        context=[db_id]
    )
    
    metrica_de_execucao = ExecutionAccuracyMetric(threshold=1.0)
    
    metrica_de_execucao.measure(caso_de_teste)
    
    assert metrica_de_execucao.is_successful(), metrica_de_execucao.reason