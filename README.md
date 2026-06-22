# MVP – Machine Learning & Analytics
## Risco de Crédito – Predição da Probabilidade de Default (PD)

## Link de execução do Colab

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleysantiago606/MVP-Machine-Learning-Analytics-PUCRIO/blob/main/MVP_Machine_Learning_Wesley_Santiago.ipynb)

Projeto desenvolvido como MVP da Pós-Graduação em Ciência de Dados e Analytics da PUC-Rio, referente à disciplina de Machine Learning & Analytics.

**Autor:** Wesley Ramos Neres Santiago
**Matrícula:** 4052025002507

---

## Objetivo

Construir e avaliar um modelo de **classificação binária** para estimar a **probabilidade de inadimplência (Probability of Default — PD)** de operações de crédito ao consumidor, a partir de atributos cadastrais do solicitante e características do empréstimo. O foco não é apenas a classe prevista, mas a **probabilidade** de *default* — insumo direto para políticas de concessão, precificação por risco e definição de limites.

Este trabalho dá continuidade ao MVP de Análise Exploratória e Pré-processamento, avançando para a etapa de modelagem preditiva.

---

## Dataset

- **Nome:** Credit Risk Dataset
- **Fonte:** [Kaggle — laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Carga:** via **URL pública** do GitHub (`pd.read_csv`), sem upload manual, login ou token — o notebook executa de ponta a ponta no Colab
- **Instâncias:** 32.581 (32.409 após a limpeza estrutural)
- **Atributos:** 12 (11 preditores + 1 variável-alvo: `loan_status`)

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| `person_age` | Numérico | Idade do solicitante em anos |
| `person_income` | Numérico | Renda anual declarada (USD) |
| `person_home_ownership` | Categórico | Situação de moradia: RENT, OWN, MORTGAGE, OTHER |
| `person_emp_length` | Numérico | Tempo de emprego em anos |
| `loan_intent` | Categórico | Finalidade do empréstimo |
| `loan_grade` | Ordinal | Classificação de risco: A (menor) → G (maior) |
| `loan_amnt` | Numérico | Valor do empréstimo solicitado (USD) |
| `loan_int_rate` | Numérico | Taxa de juros anual (%) |
| `loan_status` | Binário | **Variável-alvo**: 0 = adimplente, 1 = inadimplente |
| `loan_percent_income` | Numérico | Proporção do empréstimo sobre a renda (DTI) |
| `cb_person_default_on_file` | Binário | Histórico de default anterior: Y/N |
| `cb_person_cred_hist_length` | Numérico | Tempo de histórico de crédito em anos |

---

## Estrutura do Repositório

```
├── MVP_Machine_Learning_Wesley_Santiago.ipynb
├── credit_risk_dataset.csv
└── README.md
```

> O dataset é carregado automaticamente a partir da URL pública do próprio repositório — não é necessário baixar o arquivo manualmente nem autenticar.

---

## Etapas Realizadas no Notebook

1. **Definição do problema** — contexto, objetivo, tipo de tarefa (classificação binária para PD), premissas, hipóteses e restrições
2. **Ambiente e reprodutibilidade** — bibliotecas, fixação de semente (`SEED = 42`) e funções auxiliares de avaliação
3. **Seleção e carga dos dados** — carga por URL pública, visão geral (shape, tipos, ausentes, duplicatas) e dicionário de dados
4. **Análise exploratória (EDA)** — distribuição do alvo, qualidade dos dados, taxa de default por categoria e matriz de correlação, conectando os achados à modelagem
5. **Preparação e divisão** — remoção de registros impossíveis e duplicatas; divisão treino/teste 80/20 estratificada
6. **Pré-processamento e pipeline** — imputação, padronização e *one-hot* encapsulados em `Pipeline`/`ColumnTransformer` (sem vazamento)
7. **Baseline e modelos candidatos** — `DummyClassifier` + Regressão Logística, Random Forest, HistGradientBoosting e XGBoost
8. **Treinamento e avaliação inicial** — tabela comparativa de métricas e diagnóstico de overfitting (treino × teste)
9. **Otimização de hiperparâmetros** — `RandomizedSearchCV` com validação cruzada estratificada, regularizando o XGBoost
10. **Avaliação final** — otimização do limiar de decisão, matriz de confusão, curvas ROC/PR e curva de calibração
11. **Comparação final dos modelos** — tabela e gráfico consolidados
12. **Boas práticas e rastreabilidade** — versões de bibliotecas, semente e custo computacional
13. **Conclusão** — síntese da solução, limitações e próximos passos

---

## Como Executar

1. Acesse o notebook no Google Colab pelo badge acima
2. Execute todas as células em ordem: `Runtime → Run all`
3. A célula de carga de dados baixa o dataset automaticamente da URL pública — nenhum arquivo local é necessário
4. Nenhuma instalação adicional é necessária — todas as bibliotecas (incluindo o XGBoost) já estão disponíveis por padrão no Colab

> A etapa mais custosa é a busca de hiperparâmetros (~2 a 4 minutos no Colab, conforme a máquina alocada), controlável pelo parâmetro `N_ITER`.

---

## Bibliotecas Utilizadas

| Biblioteca | Versão recomendada | Uso |
|------------|--------------------|-----|
| `pandas` | ≥ 1.5 | Manipulação de dados |
| `numpy` | ≥ 1.23 | Operações numéricas |
| `matplotlib` | ≥ 3.6 | Visualizações |
| `scipy` | ≥ 1.9 | Distribuições para a busca aleatória e estatística KS |
| `scikit-learn` | ≥ 1.3 | Pipelines, pré-processamento, modelos e métricas |
| `xgboost` | ≥ 1.7 | Modelo de *gradient boosting* (campeão) |

---

## Modelos Avaliados

| Papel | Modelo | Origem |
|-------|--------|--------|
| Baseline | `DummyClassifier` (classe majoritária) | template |
| Candidato | Regressão Logística | template |
| Candidato | Random Forest | template |
| Candidato | **HistGradientBoosting** | **além do template** |
| Candidato | **XGBoost** | **além do template** |

A métrica principal de seleção foi a **PR-AUC**, adequada à classe minoritária sob desbalanceamento, complementada por ROC-AUC, KS, *recall*/precisão do *default* e *Brier score* (calibração).

---

## Resultados Principais

- **Limpeza:** 172 registros removidos (~0,53%) — 7 fisicamente impossíveis e 165 duplicatas exatas
- **Desbalanceamento confirmado:** 78,2% adimplentes vs 21,8% inadimplentes (ratio 3,6:1), tratado com estratificação e ponderação de classes
- **Hipótese da EDA confirmada:** as relações não-lineares e interações favorecem árvores e *boosting* sobre o modelo linear (a Regressão Logística apresentou *underfitting*)
- **Modelo campeão:** **XGBoost otimizado** — líder em discriminação
- **Otimização:** o ajuste de hiperparâmetros trouxe ganho modesto na métrica de teste, mas **reduziu o sobreajuste** (folga treino-teste de ~0,088 para ~0,050)
- **Limiar de decisão** tratado como alavanca de negócio: o ponto F1-ótimo privilegia a precisão, enquanto um corte sensível ao custo eleva o *recall*, capturando mais inadimplentes
- **Calibração:** identificado o compromisso entre discriminação e calibração — a ponderação de classes melhora o *recall*, mas infla as probabilidades na faixa intermediária

### Desempenho no Conjunto de Teste (limiar 0,5)

| Modelo | PR-AUC | ROC-AUC | KS | Brier |
|--------|:------:|:-------:|:--:|:-----:|
| Baseline (DummyClassifier) | 0,22 | 0,50 | 0,00 | 0,22 |
| Regressão Logística | 0,71 | 0,87 | 0,60 | 0,14 |
| Random Forest | 0,89 | 0,93 | 0,74 | 0,06 |
| HistGradientBoosting | 0,91 | 0,95 | 0,76 | 0,05 |
| XGBoost | 0,91 | 0,95 | 0,77 | 0,05 |
| **XGBoost (otimizado)** | **0,91** | **0,95** | **0,77** | 0,06 |

> Os dois modelos incorporados além do template (HistGradientBoosting e XGBoost) ocupam o topo do ranking. O XGBoost otimizado foi escolhido pelo melhor poder de ordenação de risco; caso a prioridade fosse a calibração das probabilidades, o HistGradientBoosting (menor Brier) seria uma alternativa igualmente defensável.

---

## Contexto Acadêmico

Este trabalho aborda a estimação da **Probabilidade de Default (PD)**, um dos três componentes do modelo IRB (*Internal Ratings-Based*) dos Acordos de Basileia II/III, ao lado da LGD (*Loss Given Default*) e da EAD (*Exposure at Default*).
