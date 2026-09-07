# Detecção de Fraudes em Transações Financeiras

Modelo de classificação que identifica transações potencialmente fraudulentas a partir de valor, horário e distância percorrida.

### O problema

Fraude em meio de pagamento é um evento raro: em uma base real, menos de 1% das transações são fraudulentas. Isso cria uma armadilha, um modelo que simplesmente responde "não é fraude" para tudo acerta 99% das vezes e não serve para nada.

O desafio, portanto, não é acertar muito. É encontrar as poucas fraudes sem acusar transações legítimas, já que cada falso positivo significa um cliente com o cartão bloqueado indevidamente.

### A abordagem
- Geração da base — transações sintéticas com três variáveis: valor, hora do dia e distância em relação ao padrão do cliente
- Análise exploratória — visualização da distribuição entre transações legítimas e fraudulentas
- Balanceamento por undersampling — redução da classe majoritária até igualar as classes, para o modelo não aprender a simplesmente ignorar as fraudes
- Treino — Random Forest com divisão estratificada em 75% treino / 25% teste
- Avaliação — matriz de confusão e relatório de precisão, recall e F1

### Resultados

### Sobre os dados

A base é sintética, gerada com NumPy para fins de estudo. A regra que define fraude no conjunto é conhecida (distância acima de 60 km combinada com horário anterior às 6h, mais 2% de ruído aleatório), o que permite verificar se o modelo consegue recuperá-la.

### Stack

Python · pandas · NumPy · scikit-learn · seaborn · matplotlib
