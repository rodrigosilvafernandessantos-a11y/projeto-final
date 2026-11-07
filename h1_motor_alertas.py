"""H1 - Monitoramento contínuo (motor de alertas) versus revisão trimestral.

H1: o monitoramento contínuo do benefício realizado reduz em pelo menos 50% o tempo
mediano de detecção de desvios em relação à revisão trimestral, com taxa de falso
alarme de até 10%.

Base sintética: carteira de iniciativas de IA acompanhadas semanalmente por 52 semanas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

rng = np.random.default_rng(42)
N_INICIATIVAS, SEMANAS = 1_000, 52
P_EVENTO = 0.40          # parcela das iniciativas que sofre queda de benefício
LIMIAR = 0.85            # alerta quando o realizado fica abaixo de 85% do prometido
JANELA, CONSECUTIVAS = 4, 2

# 1. GERAR A CARTEIRA SINTÉTICA
prometido = rng.lognormal(mean=np.log(20_000), sigma=0.6, size=N_INICIATIVAS)  # R$ por semana
nivel = rng.uniform(0.95, 1.10, N_INICIATIVAS)        # aderência inicial ao prometido
evento = rng.random(N_INICIATIVAS) < P_EVENTO
semana_evento = np.where(evento, rng.integers(5, 41, N_INICIATIVAS), SEMANAS + 1)
queda = np.where(evento, rng.uniform(0.20, 0.50, N_INICIATIVAS), 0.0)

semanas = np.arange(1, SEMANAS + 1)
fator = np.where(semanas[None, :] >= semana_evento[:, None], 1 - queda[:, None], 1.0)
ruido = rng.normal(0, 0.15, (N_INICIATIVAS, SEMANAS))
razao = np.clip(nivel[:, None] * fator + ruido, 0, None)   # realizado / prometido
realizado = razao * prometido[:, None]

base = pd.DataFrame({"iniciativa": np.repeat(np.arange(N_INICIATIVAS), SEMANAS),
                     "semana": np.tile(semanas, N_INICIATIVAS),
                     "prometido": np.repeat(prometido, SEMANAS),
                     "realizado": realizado.ravel()})
base.to_csv("carteira_sintetica.csv", index=False)

# 2. REGRAS DE DETECÇÃO
def alerta_trimestral(r):
    for fim in (13, 26, 39, 52):
        if r[fim - 13:fim].mean() < LIMIAR:
            return fim
    return None

def alerta_continuo(r):
    media = pd.Series(r).rolling(JANELA).mean().to_numpy()
    abaixo = 0
    for t in range(SEMANAS):
        abaixo = abaixo + 1 if media[t] < LIMIAR else 0
        if abaixo >= CONSECUTIVAS:
            return t + 1
    return None

# 3. AVALIAÇÃO
linhas = []
for i in range(N_INICIATIVAS):
    for metodo, regra in (("Revisão trimestral", alerta_trimestral), ("Monitoramento contínuo", alerta_continuo)):
        s = regra(razao[i])
        tau = semana_evento[i]
        falso = s is not None and (not evento[i] or s < tau)
        detectou = evento[i] and s is not None and s >= tau
        fim_perda = s if detectou else SEMANAS
        perda = (prometido[i] * nivel[i] * queda[i] * max(0, fim_perda - tau + 1)) if evento[i] else 0.0
        linhas.append(dict(iniciativa=i, metodo=metodo, evento=evento[i], semana_alerta=s,
                           falso_alarme=falso, detectou=detectou,
                           atraso=(s - tau) if detectou else np.nan, perda=perda))
av = pd.DataFrame(linhas)

print(f"Iniciativas: {N_INICIATIVAS} | com queda de benefício: {evento.sum()}")
resumo = []
for metodo, g in av.groupby("metodo", sort=False):
    ev, sem = g[g.evento], g[~g.evento]
    resumo.append(dict(metodo=metodo,
                       taxa_deteccao=ev.detectou.mean(),
                       atraso_mediano=ev.atraso.median(),
                       atraso_medio=ev.atraso.mean(),
                       falso_alarme=sem.falso_alarme.mean(),
                       perda_mi=ev.perda.sum() / 1e6))
resumo = pd.DataFrame(resumo)
pd.set_option("display.width", 120)
print(resumo.round(3).to_string(index=False))

a_tri = av[(av.metodo == "Revisão trimestral") & av.detectou].atraso
a_con = av[(av.metodo == "Monitoramento contínuo") & av.detectou].atraso
stat, p = mannwhitneyu(a_con, a_tri, alternative="less")
reducao = 1 - a_con.median() / a_tri.median()
fa = resumo.set_index("metodo").loc["Monitoramento contínuo", "falso_alarme"]
print(f"\nRedução do atraso mediano: {reducao:.0%} | Mann-Whitney U = {stat:.0f}, p = {p:.2e}")
print("H1 suportada" if reducao >= 0.5 and fa <= 0.10 and p < 0.05 else "H1 não suportada")

# 4. GRÁFICO
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": CINZA,
                     "xtick.color": CINZA, "ytick.color": CINZA})
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.arange(0, 40, 1)
ax.hist(a_tri, bins=bins, color=LARANJA, alpha=0.85, edgecolor="white", linewidth=0.6,
        label=f"Revisão trimestral (mediana {a_tri.median():.0f} semanas)")
ax.hist(a_con, bins=bins, color=AZUL, alpha=0.85, edgecolor="white", linewidth=0.6,
        label=f"Monitoramento contínuo (mediana {a_con.median():.0f} semanas)")
ax.set_xlabel("Semanas entre a queda de benefício e o alerta")
ax.set_ylabel("Número de iniciativas")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("grafico_h1_deteccao.png", dpi=200)

# 5. ANÁLISE EXPLORATÓRIA - CALIBRAÇÃO DA REGRA (não altera o veredito de H1)
print("\nCalibração exploratória da regra contínua:")
print("limiar  semanas_consecutivas  falso_alarme  atraso_mediano  taxa_deteccao")
for lim in (0.85, 0.80):
    for cons in (2, 3, 4):
        LIMIAR, CONSECUTIVAS = lim, cons
        fa_n, atrasos, det = 0, [], 0
        for i in range(N_INICIATIVAS):
            s = alerta_continuo(razao[i])
            tau = semana_evento[i]
            if s is not None and (not evento[i] or s < tau):
                fa_n += not evento[i]
            if evento[i] and s is not None and s >= tau:
                det += 1
                atrasos.append(s - tau)
        print(f"{lim:.2f}   {cons}                     {fa_n/(~evento).sum():.1%}         "
              f"{np.median(atrasos):.0f}               {det/evento.sum():.1%}")
