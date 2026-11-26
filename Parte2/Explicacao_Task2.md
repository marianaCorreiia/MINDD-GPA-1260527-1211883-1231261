# Task 2 — Modelos estatísticos de previsão (explicação)

Notebook: [task2.ipynb](./task2.ipynb) · Resultados exportados: [task2_outputs/](./task2_outputs)

## 1. O que a Task 2 pede

Prever o **número de sessões de carregamento que começam em cada uma das próximas 24 horas**, para a **Rede** (total) e para cada um dos 3 **distritos** (Nanhu, Tongxiang, Xiuzhou), usando **modelos estatísticos** de séries temporais (ETS, ARIMA/SARIMA, SARIMAX), e compará-los com as *baselines* da Task 1.

Entregável: comparação com as baselines, pontos fortes e limitações, diagnóstico de resíduos, estabilidade no tempo e entre níveis, e discussão da dupla sazonalidade (diária e semanal).

## 2. Enquadramento herdado da Task 1 (não se altera)

| Elemento | Decisão |
|---|---|
| Série | sessões iniciadas por hora, 2020-01-01 a 2021-12-31 (17 544 horas); Rede = soma dos 3 distritos |
| Emissão | diária às 00:00; horizontes h = 1…24 são as 24 horas do dia D |
| Informação | só a procura até às 23:00 do dia anterior, mais calendário |
| Estratégia | **iterada**: parâmetros estimados **uma vez por fold**; em cada origem o estado atualiza-se com os dados novos, **sem reestimar** |
| Validação | 4 *folds* trimestrais de janela expansiva (2020-Q4 a 2021-Q3) |
| Teste | 2021-Q4, **não usado** na Task 2 (o motor recusa blocos que o toquem) |
| Métricas | MAE, RMSE, MASE (m = 24); pico (acerto da hora e erro do valor); custo |
| Referência a bater | MédiaHora-7d (média da mesma hora nos últimos 7 dias) |
| Regra de seleção | MASE médio menor, melhor em ≥ 3 de 4 *folds* e IC 95% da diferença pareada de MASE diário a excluir zero |

**MASE** = MAE do modelo ÷ MAE *in-sample* do seasonal naïve de 24 h. Abaixo de 1 é melhor do que "repetir o dia anterior"; permite comparar níveis de volumes diferentes.

## 3. Estrutura do notebook

| Secção | Conteúdo |
|---|---|
| 0 | Leitura dos artefactos da Task 1 (`hourly_series_task1.csv`, `daily_weather_task1.csv`, `protocol_task1.json`) |
| 1 | **Motor de avaliação comum**: interface `fit` / `predict`, métricas, verificação (reproduz exatamente os números das baselines da Task 1) |
| 2 | Decisões transversais: transformação e janela de treino |
| 3 | Suavização exponencial: ETS, ETS semanal (m = 168), MSTL + ETS, diagnóstico de resíduos |
| 4 | Identificação ARIMA: estacionariedade, ACF/PACF, grelha de ordens |
| 5 | SARIMA, variantes para o ciclo semanal, MSTL + ARIMA, diagnóstico de resíduos |
| 6 | SARIMAX com calendário e meteorologia |
| 7 | Comparação global, seleção por nível, horizontes, estabilidade, coerência rede/distritos, custo |
| 8 | Discussão transversal |
| 9 | Exportação e síntese |

### Como o motor evita fugas de informação

O `predict` só recebe a série **já cortada** em T0, por isso um modelo não consegue ver o dia previsto nem o futuro por construção. Todos os modelos são avaliados nas mesmas origens e com as mesmas métricas, e as baselines reescritas neste motor dão exatamente os valores da Task 1 (asserções no notebook).

## 4. Decisões transversais (secção 2)

1. **Transformação `log1p`.** O erro cresce com o nível da procura; o `log1p` aproxima a variância de constante. Com ele, o MASE baixa em todos os níveis e janelas (Rede: 0,783 contra ~0,84). Sem ele, o ETS nem bate a referência na Rede. Usa-se `log1p` e não `log` porque há horas com zero sessões.
2. **Janela de treino desde 2020-04-01**, excluindo o colapso de COVID-19/Festival da Primavera. A janela quase não altera o resultado (diferença < 0,003), mas é coerente com a Task 1.
3. **Previsão = mediana** (`expm1` sem correção de enviesamento). A correção de média testada piorou MAE e MASE. Consequência: as previsões subestimam ligeiramente o volume.

## 5. Os modelos

### Suavização exponencial (secção 3)
Prevê com uma média ponderada do passado (pesos exponenciais), com componentes de nível, tendência e sazonalidade.
- **ETS(A,N,A), m = 24**: o melhor ETS. A tendência é nula à escala horária (β ≈ 0) e o AICc escolhe-o nos 16 casos.
- **ETS(A,N,A), m = 168** e **MSTL + ETS**: tentativas de incluir o ciclo semanal. Pioram: cada fator semanal só atualiza uma vez por semana, e a MSTL sobreajusta (resíduos com autocorrelação −0,25 a 24 h).

### ARIMA / SARIMA (secções 4 e 5)
Torna a série estacionária por diferenciação e modela a autocorrelação restante (AR + MA).
- A série `log1p` não é estacionária, mas **w = z(t) − z(t−24) é** (ADF e KPSS) → diferenciação sazonal D = 1, m = 24, sem segunda diferença.
- A ACF de w tem um **pico negativo isolado no lag 24** (≈ −0,48) e a PACF decai nos múltiplos de 24: é a assinatura de um **MA sazonal Q = 1**. Os lags 168 e 336 estão dentro da banda, ou seja, não há autocorrelação semanal depois de tirar o ciclo diário.
- A ordem (p, q, P, Q) identifica-se **só com o treino do F1** (grelha por AICc/BIC); a escolha entre candidatas faz-se depois na validação.
- Todo o ganho vem do MA sazonal (a ablação sem ele é igual ao seasonal naïve). O ARIMA sem sazonalidade falha por completo (MASE 2,18 na Rede).
- **SARIMA com D = 168** e **dupla diferenciação (24 e 168)** ficam pior do que repetir o dia anterior (MASE ≈ 1,06). **MSTL + ARIMA** também não bate a referência.

### SARIMAX (secção 6)
SARIMA com regressão em variáveis exógenas (erros ARMA sazonais), com as exógenas diferenciadas da mesma forma.

| Variável | Disponível em T0? | Uso |
|---|---|---|
| Calendário: fim de semana, feriado, véspera, dia seguinte | sim (futuro conhecido) | usado |
| Meteorologia do dia D−1 | hipótese operacional (known_at ≤ T0) | testada |
| Meteorologia do dia D−2 | sim | sensibilidade obrigatória |
| Meteorologia do dia D | **não** | só **oráculo diagnóstico**, nunca entra na seleção |

Uma exógena só é adotada se melhorar a validação de forma estatisticamente clara.

## 6. Resultados (MASE médio nos 4 folds de validação)

| Modelo | Rede | Nanhu | Tongxiang | Xiuzhou |
|---|---|---|---|---|
| MédiaHora-7d (referência) | 0,825 | 0,639 | 0,834 | 0,890 |
| ETS(A,N,A) | 0,783 | **0,616** | 0,784 | 0,841 |
| SARIMA (D=24) | **0,774** | 0,619 | **0,779** | 0,839 |
| SARIMAX calendário | 0,776 | 0,619 | 0,782 | **0,835** |
| SARIMAX calendário + meteo D−1 | 0,796 | 0,620 | 0,790 | 0,839 |
| ETS m = 168 | 0,805 | 0,639 | 0,800 | 0,856 |
| MSTL + ETS (4 sem) | 0,847 | 0,696 | 0,860 | 0,911 |
| SARIMA D = 168 | 1,058 | 0,826 | 1,078 | 1,154 |
| ARIMA(2,1,2) sem sazonalidade | 2,182 | 0,967 | 1,849 | 1,562 |

### Modelo selecionado por nível

Regra do protocolo + **parcimónia**: uma estrutura mais complexa só substitui a anterior se for claramente melhor (IC 95% da diferença pareada a excluir zero). Ordem de complexidade: ETS → SARIMA → SARIMAX calendário → SARIMAX calendário + meteorologia.

| Nível | Selecionado | MASE | Redução vs referência |
|---|---|---|---|
| Rede | SARIMA (1,0,1)(1,0,1)₂₄ | 0,774 | 6,2% |
| Nanhu | ETS(A,N,A) | 0,616 | 3,6% |
| Tongxiang | SARIMA (1,0,1)(1,0,1)₂₄ | 0,779 | 6,6% |
| Xiuzhou | SARIMAX calendário, (1,0,3)(0,0,1)₂₄ | 0,835 | 6,2% |

Guardado em [task2_final_selection.json](./task2_outputs/task2_final_selection.json).

## 7. Principais conclusões

1. **Quase todo o ganho vem de modelar bem o ciclo diário e atualizá-lo continuamente** (ETS ou SARIMA com MA sazonal). Os modelos mais complexos acrescentam menos de 0,01 de MASE.
2. **Tendência**: inexistente à escala horária. **Ciclo semanal**: fraco (força ≈ 0,18); todas as tentativas de o modelar dentro do modelo pioraram o resultado. Os modelos do `statsmodels` só admitem um período sazonal, e usar m = 168 ou D = 168 acumula ruído.
3. **Calendário**: coeficientes estáveis e com sentido (feriado +9% a +18%, véspera +11% a +17%), mas só melhora o erro de forma clara em Xiuzhou.
4. **Meteorologia**: sem valor. Piora com atraso de 1 dia, é indistinguível com 2 dias e piora até com o oráculo. Causa provável: sinal fraco (um único valor por dia e distrito) e sobreajuste.
5. **Resíduos** do SARIMA: sem autocorrelação sazonal (ACF < 0,035 em 24, 168 e 336 h); resta autocorrelação pequena no lag 1 (≤ 0,065) e caudas pesadas. O Ljung-Box rejeita ruído branco, mas com > 13 000 observações deteta autocorrelações sem efeito prático.
6. **Estabilidade**: os modelos selecionados ganham à referência nos 4 *folds* em todos os níveis. F2 (2021-Q1) é o mais fácil e F4 (2021-Q3) o mais difícil (em Tongxiang, F1 e F4 são equivalentes). Em Nanhu, setembro de 2021 dispara o erro em **todos** os modelos, incluindo a referência, por uma subida de nível súbita (de ~92 para ~152 sessões/dia).
7. **Horizontes**: o erro relativo é baixo de madrugada e alto a meio do dia (as horas de maior procura são as mais difíceis); o ganho sobre a referência existe em quase todos os horizontes.
8. **Coerência**: a soma das previsões dos distritos é pior do que a previsão direta da Rede (MASE 0,79 contra 0,77–0,78) e subestima o dobro; para a Rede, usa-se a previsão direta.
9. **Custo**: ETS em milissegundos por ajuste; SARIMA 8–12 s por fold; SARIMAX 34–90 s por fold. Em previsão, todos ≈ 10 ms ou menos por dia.

### Limitações principais
- **Picos achatados**: o erro do valor do pico é maior do que o da referência (Rede: ~9,9 contra 8,7 sessões), por causa da previsão da mediana em `log1p` e da suavização. Pouco adequado para dimensionar capacidade de ponta.
- **Subestimação do volume** (cerca de −0,8 a −1,0 sessões/hora na Rede).
- **Feriados** continuam a ser os dias mais difíceis.
- **Mudanças de nível súbitas** (Nanhu) são absorvidas devagar.
- Intervalos de previsão baseados na normal seriam pouco fiáveis (caudas pesadas).

## 8. Ligação à Task 3

A fasquia estatística por nível é a tabela da secção 6. Os modelos de Machine Learning têm de a superar **nas mesmas origens e com as mesmas métricas**, e é aí que se pode ganhar nos picos e nos feriados (atributos de calendário, *lags*, agregados por distrito). O teste (2021-Q4) continua por usar e será avaliado uma única vez, com tudo congelado.

## 9. Como reproduzir

1. Correr `task1.ipynb` com `Charging_Data.csv` para gerar `hourly_series_task1.csv`, `daily_weather_task1.csv` e `protocol_task1.json` (os CSV estão no `.gitignore`).
2. Correr `task2.ipynb` (≈ 18 min em 12 núcleos). Escreve em `task2_outputs/`: previsões de validação por modelo (`preds_task2_*.csv`), `summary_task2_all.csv`, `fit_times_task2.csv`, `sarima_order_grid_task2.csv`, `task2_decisions.json` e `task2_final_selection.json`.
3. Dependências: `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels` (0.15), `joblib`.
