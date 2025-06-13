# 🧠 Predição de Risco de AVC com Aprendizado de Máquina

Este projeto tem como objetivo o desenvolvimento de **modelos preditivos** para estimar o risco individual de **Acidente Vascular Cerebral (AVC)**. A proposta integra dados clínicos e de estilo de vida, e aplica técnicas modernas de **aprendizado supervisionado**, **balanceamento de classes**, **pré-processamento**, **seleção de atributos** e **avaliação de desempenho**.

O estudo compara dois algoritmos principais: **Random Forest** e **Regressão Logística**, fornecendo também visualizações gráficas das importâncias das variáveis e coeficientes para interpretação dos fatores de risco.

---

## 📁 Dataset Utilizado

- **Fonte:** `healthcare-dataset-stroke-data.xls`
- **Variáveis:** Dados clínicos (idade, hipertensão, doenças cardíacas, IMC, etc.), estilo de vida (fumo, tipo de trabalho, casamento, etc.), nível de glicose, sexo, local de residência e ocorrência de AVC (`stroke` como variável-alvo).
- **Tarefa:** Classificação binária (`stroke`: 0 ou 1)

---

## 🧪 Tecnologias e Bibliotecas

- **Linguagem**: Python
- **Bibliotecas**:
  - `pandas`, `numpy` – manipulação e análise de dados
  - `matplotlib`, `seaborn` – visualização
  - `scikit-learn` – imputação, codificação, modelagem, métricas
  - `imblearn` – balanceamento de classes com SMOTE
  - `joblib` – serialização dos modelos

---

## 🛠️ Pipeline do Projeto

1. **Leitura dos dados**:
   - Arquivo CSV lido a partir do `.xls`.

2. **Limpeza de colunas**:
   - Remoção da coluna `id` se presente.

3. **Codificação de variáveis categóricas**:
   - Colunas como `gender`, `ever_married`, `work_type`, `Residence_type` e `smoking_status` foram convertidas via **one-hot encoding**, com `drop_first=True` para evitar multicolinearidade.

4. **Imputação de dados faltantes**:
   - Utilizada a estratégia da mediana (`SimpleImputer`) para variáveis contínuas como `bmi`.

5. **Separação das variáveis**:
   - `X` contém as features.
   - `y` contém o target binário (`stroke`).

6. **Balanceamento de classes**:
   - Aplicado **SMOTE (Synthetic Minority Over-sampling Technique)** para lidar com desbalanceamento da variável alvo.

7. **Padronização dos dados**:
   - `StandardScaler` aplicado para que as variáveis tenham média 0 e desvio padrão 1.

8. **Divisão dos dados**:
   - Dividido em treino e teste com `train_test_split`, proporção 80/20.

9. **Treinamento de modelos**:
   - **RandomForestClassifier** com 100 árvores.
   - **LogisticRegression** com `max_iter=1000` para garantir convergência.

10. **Avaliação de desempenho**:
    - Métricas exibidas:
      - Acurácia
      - Precisão
      - Recall
      - F1-score

11. **Interpretação de modelos**:
    - **Random Forest**:
      - Visualização da importância das variáveis com `feature_importances_`
    - **Regressão Logística**:
      - Gráfico dos coeficientes das variáveis ordenados por magnitude

12. **Exportação dos modelos treinados**:
    - Modelos salvos como arquivos `.pkl`:
      - `rf_avc_model.pkl`
      - `lr_avc_model.pkl`

---

## 📊 Casos de Visualizações

- **Importância das variáveis (Random Forest)**;
- **Coeficientes da Regressão Logística**;

> *(Os gráficos são gerados automaticamente durante a execução do código.)*

---

## 🚀 Como Executar

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
