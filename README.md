# Projeto de Fine-Tuning e Avaliação de LLM para Text-to-SQL

## Resumo do Projeto (Abstract)

Este projeto investiga o trade-off entre ganho de performance e regressão de capacidade ao especializar um Modelo de Linguagem de Grande Escala (LLM) para a tarefa de tradução de texto para SQL (Text-to-SQL). Utilizando o modelo `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` como base, duas variantes foram especializadas no dataset Spider através da técnica de Low-Rank Adaptation (LoRA). A performance foi medida com uma metodologia de avaliação dupla: (1) uma métrica customizada de Acurácia de Execução para a tarefa Text-to-SQL e (2) uma avaliação de conhecimento geral no benchmark MMLU. Os resultados demonstram um ganho substancial na tarefa de especialização, ao custo de uma modesta e não uniforme regressão de capacidade geral, destacando a eficácia da abordagem e a necessidade de pipelines de avaliação holística.

## Estrutura do Projeto

Para executar este projeto, os arquivos devem estar organizados da seguinte forma:

```
/MESTRADO/
├── database/               # Pasta com os bancos de dados do Spider (do database.zip)
├── lora_model_version1/    # Adaptador do Modelo 1 (gerado pelo fine-tuning)
├── lora_model_version2/    # Adaptador do Modelo 2 (gerado pelo fine-tuning)
|
├── dev.json                # Split de desenvolvimento do Spider
├── train_spider.json       # Split de treino do Spider
├── economy.parquet         # Dados do MMLU (Ciências Sociais)
├── philosophy.parquet      # Dados do MMLU (Humanidades)
├── computer_science.parquet# Dados do MMLU (STEM)
|
├── TP4_finetunes.ipynb     # Notebook para preparação de dados e fine-tuning
├── TP4_Baseline_execution.ipynb # Notebook para avaliação do modelo base
├── test_evaluation.py      # Script de avaliação Text-to-SQL (Fase 3)
├── execution_metric.py     # Métrica customizada para a avaliação SQL
├── mmlu_avaliator.py       # Script de avaliação MMLU (Fase 4)
|
├── prompt_template.txt     # Template do prompt few-shot para SQL
└── requirements.txt        # Dependências do projeto
```

## Pré-requisitos

1.  **Ambiente Python:** Recomenda-se o uso de `conda` para gerenciar o ambiente. Python 3.11 ou superior.
2.  **Hardware:** Acesso a uma GPU NVIDIA com CUDA instalado. O treinamento e a avaliação foram conduzidos em uma Tesla T4.
3.  **Dataset Spider:** É necessário baixar o dataset Spider completo, incluindo os arquivos `train_spider.json`, `dev.json` e, crucialmente, descompactar o `database.zip` na raiz do projeto.

## Instalação

Todo o ambiente de software pode ser configurado a partir do arquivo `requirements.txt`.

1.  **Crie e ative um novo ambiente Conda:**
    ```bash
    conda create -n llm_evaluate python=3.11 -y
    conda activate llm_evaluate
    ```

2.  **Instale todas as dependências:**
    O comando a seguir instalará o `unsloth` (que por sua vez instala `torch`, `bitsandbytes`, `xformers`, etc.) e as bibliotecas de avaliação.

    ```bash
    pip install -r requirements.txt
    ```

## Configuração

Antes de executar os scripts de avaliação, é **essencial** verificar e ajustar os caminhos dos arquivos e modelos.

1.  **No arquivo `test_evaluation.py`:**
    * Verifique se as variáveis `PATH_ADAPTADOR_1` e `PATH_ADAPTADOR_2` apontam para os diretórios corretos onde os modelos fine-tuned foram salvos.
    ```python
    PATH_ADAPTADOR_1 = "lora_model_version1"
    PATH_ADAPTADOR_2 = "lora_model_version2" 
    ```

2.  **No arquivo `mmlu_avaliator.py`:**
    * Similarmente, verifique os caminhos `PATH_ADAPTADOR_1` e `PATH_ADAPTADOR_2`.
    * Certifique-se de que os nomes dos arquivos em `MMLU_FILES` correspondem aos arquivos que você baixou.
    ```python
    PATH_ADAPTADOR_1 = "lora_model_version1"
    PATH_ADAPTADOR_2 = "lora_model_version2" 
    
    MMLU_FILES = {
        "Ciências Sociais": "economy.parquet",
        "Humanidades": "philosophy.parquet", # ATENÇÃO: nome do arquivo no seu PC
        "STEM": "computer_science.parquet",    # ATENÇÃO: nome do arquivo no seu PC
    }
    ```

3.  **Nos Notebooks (`.ipynb`):**
    * Todos os caminhos de arquivos dentro dos notebooks (ex: `/dados/dev.json`, `/content/drive/...`) devem ser ajustados para a estrutura de pastas local do seu projeto.

## Executando os Experimentos

O processo é dividido em fases, seguindo a execução dos notebooks e scripts.

1.  **Fase de Fine-Tuning:**
    * Execute as células do notebook `TP4_finetunes.ipynb` para preparar os dados e treinar os dois modelos. Isso irá gerar as pastas `lora_model_version1` e `lora_model_version2`.

2.  **Avaliação do Baseline (SQL):**
    * Execute as células do notebook `TP4_Baseline_execution.ipynb` para obter a performance do modelo base.

3.  **Avaliação dos Modelos Fine-Tuned (SQL - Fase 3):**
    * No terminal, com o ambiente `llm_evaluate` ativo, execute o `pytest`:
    ```bash
    pytest test_evaluation.py --html=relatorio_sql.html --self-contained-html
    ```

4.  **Avaliação de Regressão (MMLU - Fase 4):**
    * No terminal, execute o script Python:
    ```bash
    python mmlu_avaliator.py
    ```

## Resultados Esperados

* A execução do `pytest` gerará o arquivo `relatorio_sql.html`, contendo o detalhe de cada sucesso e falha na tarefa Text-to-SQL.
* A execução do `mmlu_avaliator.py` imprimirá no terminal a tabela final de "Análise Quantitativa de Regressão de Capacidade", comparando a acurácia dos modelos.

---
### Conteúdo para o `requirements.txt`

Crie um arquivo chamado `requirements.txt` e cole o seguinte conteúdo nele. Este arquivo consolida todas as dependências necessárias.

```
# Instalação principal do Unsloth com dependências de GPU
unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)

# Ferramentas de avaliação
deepeval
pytest
pytest-html

# Manipulação de dados
pandas
pyarrow
```
