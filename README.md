# Liber_Capital_Case
Solução do Case como parte do processo seletivo para Liber Capital.

## Autor

- [Franklin Oliveira](https://www.linkedin.com/in/franklin-oliveira95/)

## Base de dados

A base de dados trabalhada é a [Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29), disponível no [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 

Nesse *dataset*, indivíduos são classificados como "bons" ou "ruins" de acordo com seu nível de risco de crédito, estimado com base em um conjunto de atributos.  

## Introdução

O case visa a elaboração de uma análise preditiva de classificação de clientes solicitantes, avaliados pelo seu risco de crédito (`1 = bom, 2 = ruim`). Portanto, trata-se de um problema de classificação onde serão empregados modelos de aprendizado supervisionado a fim de automatizar o processo decisório de concessão de crédito, buscando minimizar o risco de inadimplência.

<p color='red'>Incluir breve descrição das análises que fiz</p>

## Estrutura

<p color='red'>Adicionar novos arquivos</p>

```bash
.
├── data
│   ├── data.pickle                     # versão pré-processada da base de dados
│   ├── german.data                     # base de dados (raw)
│   ├── german.doc                      # descrição da base de dados
├── models                              # arquivos dos modelos salvos
├── notebooks
│   ├── 00_preprocessing_and_feature_engineering.ipynb
│   ├── 01_eda.ipynb
├── README.md
├── requirements.txt
└── Teste Cientista de Dados pl.pdf     # descrição do case em PDF
```

## Requerimentos

Principais libraries:

```sh
numpy - pandas - swifter - seaborn - matplotlib - statsmodels 
```

Todas as bibliotecas necessárias para executar os arquivos de notebook (`.ipynb`) e suas dependências estão listadas no arquivo `requirements.txt`.

Para instalar todos os requerimentos necessários (libraries principais e suas dependências), basta executar o seguinte comando em uma instância de terminal (obs: certifique-se de navegar até a pasta correta):

```bash
pip install -r requirements.txt
```


## Possíveis melhorias
