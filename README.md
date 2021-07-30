# Statlog Credit Risk Analysis

Aplicando modelos de Machine Learning para classificação do risco de crédito.

<img src="./src/Credit-Scores.jpg" height=250px>

## Autor

- [Franklin Oliveira](https://www.linkedin.com/in/franklin-oliveira95/)

## Base de dados

A base de dados trabalhada é a [Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29), disponível no [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 

Nesse *dataset*, indivíduos são classificados como "bons" ou "ruins" de acordo com seu nível de risco de crédito, estimado com base em um conjunto de atributos. Ao todo, tem-se 21 informações (_features_) acerca de 1000 clientes. A imagem a seguir exibe uma prévia do conjunto de dados.

![](src/dataset_head.png)

Cada feature é identificada por sua posição, onde a última coluna representa a variável resposta (risco de crédito). Cada feature foi nomeada e decodificada segundo o dicionário presente em `./data/german.doc`.

## Introdução

O case visa a elaboração de uma análise preditiva de classificação de clientes solicitantes, avaliados pelo seu risco de crédito (`1 = bom, 2 = ruim`). Portanto, trata-se de um problema de classificação onde serão empregados modelos de aprendizado supervisionado a fim de automatizar o processo decisório de concessão de crédito, buscando minimizar o risco de inadimplência.

Nesse sentido, foi realizada uma análise exploratória de dados e um processo de *feature engineering* a fim de preparar o dataset para aplicação de modelos preditivos. Não foi necessário lidar com *missing values*. 

## Estrutura

```bash
.
├── data
│   ├── data.pickle                                   # versão pré-processada da base de dados
│   ├── german.data                                   # base de dados (raw)
│   ├── german.doc                                    # descrição da base de dados
├── models                                            # arquivos dos modelos salvos
│   ├── decision_tree.pickle                      
│   ├── decision_tree_holdout_metrics.pickle                      
│   ├── logistic_regression.pickle                      
│   ├── logistic_regression_holdout_metrics.pickle                      
│   ├── random_forest.pickle                          # arquivo do melhor modelo 
│   ├── random_forest_holdout_metrics.pickle          # métricas do melhor modelo no conjunto holdout            
├── notebooks
│   ├── 00_preprocessing_and_feature_engineering.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_decision_trees.ipynb
│   ├── 02_linear_model.ipynb
│   ├── 02_random_forests.ipynb
│   ├── 03_model_comparison.ipynb
├── src                                               # arquivos de ilustração para o README 
├── README.md
├── requirements.txt        
└── proposta.pdf                                      # descrição do desafio em PDF
```

## Requerimentos

Todo esse estudo foi desenvolvido no sistema operacional Ubuntu 20.04 LTS, com Python versão 3.8 em uma distribuição Anaconda. No entanto, tomou-se o devido cuidado para que os arquivos notebook sejam executados também em sistemas Windows e Mac OS com Python 3.X instalado, desde que sejam atendidos os requisitos de libraries listados a seguir.

Principais libraries:

```sh
numpy - pandas - swifter - seaborn - matplotlib - statsmodels - sklearn 
```

Todas as bibliotecas necessárias para executar os arquivos de notebook (`.ipynb`) e suas dependências estão listadas no arquivo `requirements.txt`.

Para instalar todos os requerimentos necessários (libraries principais e suas dependências), basta executar o seguinte comando em uma instância de terminal (obs: certifique-se de navegar até a pasta correta):

```bash
pip install -r requirements.txt
```


## Possíveis melhorias

- Converter blocos de código reutilizado em funções. Criar um script `.py` e importá-las nos arquivos _notebook_ `.ipynb` quando necessárias. 
- Desenvolver novas _features_ combinando uma ou mais informações presentes no dataset e avaliar se esse mecanismo melhora a performance dos modelos preditivos.
- Estimar um modelo de _boosting_ (ex.: XGBoost) e comparar sua performance com a dos demais modelos.
- Criar pipeline incorporando as etapas de *feature engineering*, treinamento e previsão com `Scikit-Learn` para inserir em ambientes de produção.


## Desempenho dos modelos

Nesse estudo, foram avaliadas três classes de modelos para a tarefa de classificação: Regressão Logística, Árvores de Decisão e Random Forest. A figura abaixo ilustra as métricas de performance obtidas: 

![](src/holdout_performance_comparison.png)

![](src/time_comparison.png)