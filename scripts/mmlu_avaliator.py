import json
import os
import torch
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm

# --- 1. CONFIGURAÇÃO ---
BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
PATH_ADAPTADOR_1 = "lora_model_version1"
PATH_ADAPTADOR_2 = "lora_model_version2" 

MMLU_FILES = {
    "Ciências Sociais": "economy.parquet",
    "Humanidades": "philosophia.parquet",
    "STEM": "computer.parquet",
}
NUM_QUESTIONS_PER_CATEGORY = 50

FOUR_SHOT_EXAMPLES = """Question: Which element has the chemical symbol 'O'?
Answer: Oxygen

Question: Who wrote "Hamlet"?
Answer: William Shakespeare

Question: Solve for x: 2x + 3 = 7.
Answer: 2

Question: Which planet is known as the Red Planet?
Answer: Mars
"""

# --- 2. FUNÇÕES AUXILIARES ---

def load_lora_model(model_path):
    """Carrega um modelo base com um adaptador LoRA ou apenas um modelo base."""
    print(f"Carregando modelo de: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"Modelo '{model_path}' carregado.")
    return model, tokenizer

def load_mmlu_data(file_path, num_questions):
    """Carrega e limita o número de perguntas de um arquivo MMLU."""
    df = pd.read_parquet(file_path)
    # Adicionamos um print para verificar o tamanho real de cada arquivo
    print(f"  -> Arquivo '{file_path}' contém {len(df)} perguntas. Usando as primeiras {min(len(df), num_questions)}.")
    df = df.head(num_questions)
    
    questions = []
    for _, row in df.iterrows():
        choices = {chr(65 + i): choice for i, choice in enumerate(row['choices'])}
        questions.append({
            "question": row['question'],
            "choices": choices,
            "answer": chr(65 + row['answer']),
        })
    return questions

def get_log_likelihood(model, tokenizer, text):
    """Calcula a log-likelihood de uma sequência de texto."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = -outputs.loss.item()
    return log_likelihood

def evaluate_model_on_mmlu(model, tokenizer, category, file_path, num_questions):
    """Função dedicada a avaliar um único modelo em uma única categoria."""
    print(f"\nAvaliando categoria: {category}...")
    questions = load_mmlu_data(file_path, num_questions)
    correct_count = 0
    
    for question_data in tqdm(questions, desc=f"  Progresso {category}"):
        prompt_base = f"{FOUR_SHOT_EXAMPLES}\nPergunta: {question_data['question']}\nResposta:"
        
        log_likelihoods = {}
        for option_key, option_text in question_data['choices'].items():
            full_prompt = f"{prompt_base} {option_text}"
            log_likelihoods[option_key] = get_log_likelihood(model, tokenizer, full_prompt)
        
        model_answer = max(log_likelihoods, key=log_likelihoods.get)
        
        if model_answer == question_data['answer']:
            correct_count += 1
            
    accuracy = (correct_count / len(questions)) * 100 if questions else 0
    return accuracy

# --- 3. ANÁLISE E RELATÓRIO ---

def analyze_and_report_results(results):
    print("\n\n--- ANÁLISE QUANTITATIVA DE REGRESSÃO DE CAPACIDADE ---")
    
    base_results = results.get("Base", {})
    
    for model_name, category_results in results.items():
        if model_name == "Base":
            print(f"\n--- Resultados para o Modelo: {model_name} ---")
            for category, accuracy in category_results.items():
                print(f"  Categoria: {category:<20} | Acurácia: {accuracy:.2f}%")
            continue

        print(f"\n--- Análise para o Modelo: {model_name} ---")
        
        total_correct_ft = 0
        total_correct_base = 0
        total_questions = 0
        
        for category, accuracy in category_results.items():
            base_accuracy = base_results.get(category, 0)
            num_q_in_cat = NUM_QUESTIONS_PER_CATEGORY # Simplificação, idealmente seria o len real
            
            total_correct_ft += accuracy * (num_q_in_cat / 100)
            total_correct_base += base_accuracy * (num_q_in_cat / 100)
            total_questions += num_q_in_cat
            
            regression = accuracy - base_accuracy
            print(f"  Categoria: {category:<20} | Acurácia: {accuracy:.2f}% (Variação: {regression:+.2f} pts)")
            
        if total_questions > 0:
            agg_accuracy_ft = (total_correct_ft / total_questions) * 100
            agg_accuracy_base = (total_correct_base / total_questions) * 100
            agg_regression = agg_accuracy_ft - agg_accuracy_base
            print("---------------------------------------------------------")
            print(f"  AGREGADO: {'':<20} | Acurácia: {agg_accuracy_ft:.2f}% (Variação Total: {agg_regression:+.2f} pts)")

# --- 4. EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    # Dicionário para armazenar todos os resultados
    all_results = {}

    # --- Avalia o Modelo Base ---
    print("="*50)
    print("INICIANDO AVALIAÇÃO DO MODELO BASE")
    print("="*50)
    base_model, base_tokenizer = load_lora_model(BASE_MODEL_NAME)
    all_results["Base"] = {}
    for category, file_path in MMLU_FILES.items():
        accuracy = evaluate_model_on_mmlu(base_model, base_tokenizer, category, file_path, NUM_QUESTIONS_PER_CATEGORY)
        all_results["Base"][category] = accuracy
    del base_model # Libera a memória da GPU
    torch.cuda.empty_cache()

    # --- Avalia o Modelo Fine-Tuned 1 ---
    print("="*50)
    print("INICIANDO AVALIAÇÃO DO MODELO FINE-TUNED 1")
    print("="*50)
    model_1, tokenizer_1 = load_lora_model(PATH_ADAPTADOR_1)
    all_results["Fine-Tuned 1"] = {}
    for category, file_path in MMLU_FILES.items():
        accuracy = evaluate_model_on_mmlu(model_1, tokenizer_1, category, file_path, NUM_QUESTIONS_PER_CATEGORY)
        all_results["Fine-Tuned 1"][category] = accuracy
    del model_1 # Libera a memória da GPU
    torch.cuda.empty_cache()

    # --- Avalia o Modelo Fine-Tuned 2 ---
    print("="*50)
    print("INICIANDO AVALIAÇÃO DO MODELO FINE-TUNED 2")
    print("="*50)
    model_2, tokenizer_2 = load_lora_model(PATH_ADAPTADOR_2)
    all_results["Fine-Tuned 2"] = {}
    for category, file_path in MMLU_FILES.items():
        accuracy = evaluate_model_on_mmlu(model_2, tokenizer_2, category, file_path, NUM_QUESTIONS_PER_CATEGORY)
        all_results["Fine-Tuned 2"][category] = accuracy
    del model_2 # Libera a memória da GPU
    torch.cuda.empty_cache()
    
    # --- Apresenta o Relatório Final ---
    analyze_and_report_results(all_results)