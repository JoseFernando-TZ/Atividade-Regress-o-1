import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec
from sklearn . metrics import r2_score , mean_squared_error
import numpy as np
import statsmodels . api as sm


df = pd.read_csv("C:\\Users\\Usuario\\Downloads\\Medicaldataset.csv")

#Questão 1
'''nome: heart attack dataset'''
'''fonte: kaggle'''
print(df.shape)
'''linhas: 1319 colunas: 9'''
#Questão 2
print(df.head())

#Questão 3
'''nome das variveis:'''
print(df.dtypes)
'''variaveis quantitativas: todas menos result'''
print(df.describe().T)

#variável resposta: triponina
#váriavel explicativa: age

#Questão 4
#maior media: blood sugar
#maior dispersão: blood sugar
#variavel com valores diferentes das demais: Troponin

#Questão 5
x = df['Age'].values
y = df['Troponin'].values

b1, b0, r, p_valor, se_b1 = stats.linregress(x, y)

y_pred = b0 + b1 * x
residuos = y - y_pred
r2 = r ** 2
rmse = np.sqrt(np.mean(residuos ** 2))
n = len(x)

print("=" * 50)
print("   REGRESSÃO LINEAR SIMPLES")
print("   ŷ = b0 + b1·x")
print("=" * 50)
print(f"  b0 (intercepto) = {b0:.6f}")
print(f"  b1 (slope)      = {b1:.6f}")
print(f"  Equação: ŷ = {b0:.4f} + {b1:.4f}·Age")
print(f"  R               = {r:.4f}")
print(f"  R²              = {r2:.4f}  ({r2 * 100:.2f}%)")
print(f"  p-value (b1)    = {p_valor:.6f}")
print(f"  SE (b1)         = {se_b1:.6f}")
print(f"  RMSE            = {rmse:.4f}")
print(f"  n               = {n}")
print("=" * 50)

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Regressão Linear Simples — Age × Troponin\n"
             f"ŷ = {b0:.4f} + {b1:.4f}·Age     "
             f"R² = {r2:.4f}   p = {p_valor:.4f}",
             fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colors = ['#E24B4A' if v == 'positive' else '#378ADD'
          for v in df['Result']]

print(
    "variavel y, troponina é o desfecho de interesse, ela é o principal biomarcador de indarto do miocardio ou infarto agudo.")
print("variavel x, é a preditora da triponina.")
print("\n")

print("justifcativa da escolha:")
print(
    "Para uma regressão linear simples é necessário escolher uma variável contínua com relação linear plausível com Y. " \
    "A idade atende a esse critério, ao contrário de variáveis categóricas ou binárias do banco de dados" \
    "A idade é um dos fatores de risco cardiovascular mais consolidados na literatura médica. Com o envelhecimento," \
    " ocorrem processos como arteriosclerose, rigidez arterial e declínio da função cardíaca, que aumentam a probabilidade de eventos isquêmicos," \
    " e consequentemente de elevação da Troponina.")

#Questão 6
plt.scatter(df["Age"],df["Troponin"])
plt.xlabel("Age")
plt.ylabel("Troponin")
plt.title("Grafico_de_dispersão")
plt.show()

#a) Quando a idade está entre 40-80 a troponina tende a aumentar
#b) positivo com coeficiente 0.007512 aproximadamente
#c) não é linear

#Questão 7
X = np.array(df["Age"]).reshape(-1, 1)
Y = np.array(df["Troponin"])

modelo = LinearRegression().fit(X, Y)
print(f"Coeficiente Angular (β1): {modelo.coef_[0]}")


modelo_simples = LinearRegression().fit(X, Y)
print("intercepto: ", modelo_simples.intercept_)
print("coeficiente: ", modelo_simples.coef_)

#valor do intercepto: -0.06120034318552897
#valor do coeficiente:  [0.00751253]
#equação: ŷ = -0.0612 + 0.0075·Age

#Questão 8
#o aumento do nivel da  troponina ao decorrer da idade é um claro marcador que ao decorredr do aumento da idade
#aumenta o risco do infarto agudo ouu do miocardio, e as pessoas que se mantém em um nivel baixo antes do 60
#tendem a se manter abaixo após essa idade.

#Questão 9
y_pred_simples = modelo_simples . predict ( X )

r2_simples = r2_score (y , y_pred_simples )
mse_simples = mean_squared_error (y , y_pred_simples )
rmse_simples = np . sqrt ( mse_simples )
print ("R2:", r2_simples )
print ("MSE :", mse_simples )
print (" RMSE :", rmse_simples )
pd.set_option('display.max_columns', None)


#Questão 10
# A) Ainda que a idade tenha influência na troponina falta informações para explicar a variação no gráfico
# B) Um R² de 0,0056 está muito abaixo do limiar mínimo para ser
# considerado um modelo com poder explicativo razoável

#Questão 11

# a) Variável resposta
# Troponin (mesma do modelo simples)

# b) Variáveis explicativas escolhidas

# c) Não. Um R² alto indica apenas que o modelo linear se ajusta bem aos dados
# ou seja, que há associação estatística entre as variáveis
# Correlação e causalidade são conceitos distintos. Por exemplo, duas variáveis
# podem apresentar R² elevado por coincidência (correlação espúria)
# , por influência de uma terceira variável não observada (variável de confusão)
# , ou simplesmente porque ambas crescem juntas ao longo do tempo sem qualquer relação direta.
# Para afirmar causalidade, são necessários critérios adicionais como delineamento experimental controlado
# , temporalidade e plausibilidade biológica ou teórica.

X_multiplo = df[['Age', 'Blood sugar', 'Heart rate']]
y = df['Troponin']

# c), d), e) Ajuste do modelo e exibição dos coeficientes
modelo_multiplo = LinearRegression()
modelo_multiplo.fit(X_multiplo, y)

print("Intercepto:", modelo_multiplo.intercept_)
print("Coeficientes:", modelo_multiplo.coef_)

variaveis = ['Age', 'Blood sugar', 'Heart rate']
for var, coef in zip(variaveis, modelo_multiplo.coef_):
    print(f"  {var}: {coef:.6f}")

# Questão 12
#
# a) O que significa o coeficiente da primeira variável explicativa (Age)?
# O coeficiente para Age é 0,0075. Isso significa que, para cada aumento de 1
# ano na idade do paciente, espera-se um aumento médio de 0,0075 unidades no nível de Troponina,
# desde que os níveis de açúcar no sangue e a frequência cardíaca não se alterem. Como o p-valor é 0,001,
# essa variável é estatisticamente significativa para o modelo.
#
# b) O que significa o coeficiente da segunda variável explicativa (Blood sugar)?
# O coeficiente para Blood sugar é 0,0003. Isso indica que, para cada aumento de 1 mg/dL
# (ou a unidade de medida utilizada) de açúcar no sangue, o nível de Troponina aumenta, em média,
# apenas 0,0003 unidades. Vale notar que, como o p-valor é alto (0,429), esse coeficiente não é estatisticamente significativo,
# sugerindo que o açúcar no sangue tem pouco ou nenhum impacto real na Troponina dentro deste modelo.
#
# c) Por que, na regressão múltipla, é importante dizer “mantendo as demais variáveis constantes”?
# Essa cláusula (também conhecida como ceteris paribus) é essencial porque, em um modelo múltiplo,
# as variáveis explicativas podem estar correlacionadas entre si. Ao dizer "mantendo as demais constantes",
# isolamos o efeito direto e individual de uma única variável sobre a Troponina, removendo a interferência ou a
# confusão que as outras variáveis poderiam causar no cálculo do impacto daquele fator específico.

#Questão 13

y_pred_multiplo = modelo_multiplo.predict(X_multiplo)

r2_multiplo = r2_score(y, y_pred_multiplo)
mse_multiplo = mean_squared_error(y, y_pred_multiplo)
rmse_multiplo = np.sqrt(mse_multiplo)

print("R²:", r2_multiplo)
print("MSE:", mse_multiplo)
print("RMSE:", rmse_multiplo)

#Questão 15

nova_observacao = pd.DataFrame({
    'Age': [55],
    'Blood sugar': [120],
    'Heart rate': [80]
})

previsao = modelo_multiplo.predict(nova_observacao)

print("Valor previsto de Troponina:", previsao[0])
'''
Para um paciente hipotético de 55 anos, com glicemia de 120 mg/dL e frequência cardíaca 
de 80 bpm, o modelo estima um nível de Troponina de aproximadamente 0,332. Esse valor, 
isoladamente, estaria dentro de uma faixa que pode indicar ausência de infarto agudo, 
já que níveis clinicamente preocupantes de Troponina costumam ser consideravelmente
mais elevados. Contudo, dado o R² muito baixo do modelo (≈ 0,78%), essa previsão deve ser 
interpretada com extrema cautela, pois o modelo captura uma fração muito pequena da variabilidade 
real da Troponina e não é confiável para uso preditivo clínico.
'''
#Questão 14

# a)A regressão linear múltipla (R² = 0,0085 > 0,0079).
# b)A regressão linear múltipla (MSE = 1,3207 < 1,3215).
# c)A regressão linear múltipla (RMSE = 1,1492 < 1,1496).
# d)Sim, o modelo múltiplo apresentou melhora em todos os três critérios avaliados:
# maior R² e menores MSE e RMSE.
# e)A melhora foi muito pequena e praticamente negligenciável.
# A diferença no R² foi de apenas 0,0006 (0,06 pontos percentuais),
# e as reduções em MSE e RMSE foram decimais irrisórias.
# Isso indica que as variáveis adicionadas na regressão múltipla
# (Blood sugar e Heart rate) contribuíram muito pouco para explicar a variação da Troponina,
# algo confirmado pelos p-values não significativos dessas duas variáveis (p = 0,429 e p = 0,618, respectivamente).
# Apenas a Idade permanece como preditor significativo (p = 0,001).

#Questão 15

nova_observacao = pd.DataFrame({
    'Age': [55],
    'Blood sugar': [120],
    'Heart rate': [80]
})

previsao = modelo_multiplo.predict(nova_observacao)

print("Valor previsto de Troponina:", previsao[0])
'''
Para um paciente hipotético de 55 anos, com glicemia de 120 mg/dL e frequência cardíaca 
de 80 bpm, o modelo estima um nível de Troponina de aproximadamente 0,332. Esse valor, 
isoladamente, estaria dentro de uma faixa que pode indicar ausência de infarto agudo, 
já que níveis clinicamente preocupantes de Troponina costumam ser consideravelmente
mais elevados. Contudo, dado o R² muito baixo do modelo (≈ 0,78%), essa previsão deve ser 
interpretada com extrema cautela, pois o modelo captura uma fração muito pequena da variabilidade 
real da Troponina e não é confiável para uso preditivo clínico.
'''

#Questão 16

X_sm = df[['Age', 'Blood sugar', 'Heart rate']]
X_sm = sm.add_constant(X_sm)
y = df['Troponin']

modelo_sm = sm.OLS(y, X_sm).fit()

print(modelo_sm.summary())

# a) R² = 0,009 (aproximadamente 0,85% — usando o valor mais preciso: 0,008533)
# b) R² ajustado = 0,006

#Questão 17

# O R² mede a proporção da variância da variável resposta explicada pelo modelo.
# Seu problema é que ele sempre aumenta quando novas variáveis são adicionadas,
# mesmo que essas variáveis não contribuam de forma relevante o que pode dar uma
# falsa impressão de melhora do modelo.
# O R² ajustado corrige esse comportamento ao penalizar a inclusão de variáveis
# que não melhoram o ajuste de forma significativa. Ele leva em conta o número de
# preditores e o tamanho da amostra, podendo até diminuir quando uma variável irrelevante é inserida.
# Por isso, o R² ajustado é mais adequado para avaliar regressão múltipla:
# ele permite comparar modelos com diferentes números de variáveis de forma mais justa e honesta.
# Neste caso, o R² ajustado (0,006) ser menor que o R² (0,009) já sinaliza que as variáveis Blood sugar
# e Heart rate adicionaram pouco valor real ao modelo.

# Questão 18

# a)Investigar se variáveis clínicas e demográficas são capazes de prever os níveis de Troponina,
# principal biomarcador de infarto agudo do miocárdio, por meio de modelos de regressão linear simples e múltipla.
#
# b)Uma base de dados clínicos com 1.319 observações e 9 variáveis, contendo informações como idade, gênero,
# frequência cardíaca, pressão arterial, glicemia, CK-MB, Troponina e diagnóstico (positivo/negativo para infarto).
#
# c)Troponina — biomarcador sérico utilizado clinicamente para diagnóstico de infarto agudo do miocárdio.
#
# d)
# Regressão simples: Idade (Age)
# Regressão múltipla: Idade, Glicemia (Blood sugar) e Frequência cardíaca (Heart rate)
#
# e)A regressão múltipla apresentou valores ligeiramente melhores (R² = 0,0085, MSE = 1,3207, RMSE = 1,1492),
# porém a diferença em relação ao modelo simples é mínima e praticamente desprezível.
#
# f)Não de forma confiável. Ambos os modelos apresentam R² extremamente baixo (inferior a 1%),
# o que indica que as variáveis escolhidas explicam menos de 1% da variação da Troponina. Além disso,
# o RMSE elevado (~1,15) e a forte assimetria e curtose dos resíduos (Skew = 5,755; Kurtosis = 42,563)
# evidenciam que os pressupostos da regressão linear não estão sendo atendidos. O modelo não tem capacidade
# preditiva adequada para uso clínico.
#
# g)
# Baixo poder explicativo: R² < 1% indica que variáveis cruciais para predizer Troponina
# (como oclusão coronariana, histórico de IAM, tempo de dor) não estão no modelo.
# Violação de pressupostos: A Troponina tem distribuição fortemente assimétrica
# (assimetria positiva com valores extremos), o que viola a premissa de normalidade dos resíduos da
# regressão linear.
# Variáveis descartadas: Blood sugar e Heart rate não foram estatisticamente significativas,
# sugerindo que a seleção de preditores precisa ser revista.
# Possível necessidade de transformação: Transformações como log(Troponina) poderiam melhorar o ajuste.
# Modelos alternativos: Dada a natureza do desfecho (diagnóstico positivo/negativo),
# uma regressão logística poderia ser mais apropriada para este banco de dados.