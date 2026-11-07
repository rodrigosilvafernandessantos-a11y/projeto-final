"""H2 e H3 - Simulação de Monte Carlo do modelo financeiro.

H2: a plataforma apresenta VPL positivo em pelo menos 70% dos cenários simulados.
H3: o ritmo de aquisição de clientes é a premissa de maior influência sobre o VPL.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from modelo_financeiro import projetar, BASE

N = 10_000
rng = np.random.default_rng(42)

# 1. DISTRIBUIÇÕES DAS PREMISSAS (mínimo, moda, máximo)
premissas = pd.DataFrame({
    "mult_clientes": rng.triangular(0.50, 1.00, 1.20, N),  # ritmo de aquisição
    "mult_preco":    rng.triangular(0.80, 1.00, 1.10, N),  # preço praticado
    "aceite":        rng.uniform(0.60, 0.90, N),           # taxa de aceite dos agentes
    "custo_var":     rng.triangular(0.04, 0.06, 0.12, N),  # custo de IA, voz e nuvem
    "mult_fixos":    rng.triangular(0.90, 1.00, 1.15, N),  # custos fixos
    "churn":         rng.triangular(0.00, 0.05, 0.20, N),  # rotatividade anual
})

# 2. SIMULAÇÃO
saidas = [projetar(**linha) for linha in premissas.to_dict("records")]
res = premissas.copy()
res["vpl"] = [s["vpl"] for s in saidas]
res["caixa_min"] = [s["caixa_min"] for s in saidas]
res["equilibrio"] = [s["ano_equilibrio"] or 99 for s in saidas]

# 3. H2 - PROBABILIDADE DE VPL POSITIVO
p_pos = (res["vpl"] > 0).mean()
erro = 1.96 * np.sqrt(p_pos * (1 - p_pos) / N)
p10, p50, p90 = np.percentile(res["vpl"], [10, 50, 90])
print("=== H2 ===")
print(f"P(VPL > 0) = {p_pos:.1%}  (IC 95%: {p_pos-erro:.1%} a {p_pos+erro:.1%})")
print(f"VPL P10 = {p10:.2f} | P50 = {p50:.2f} | P90 = {p90:.2f} | média = {res['vpl'].mean():.2f}")
print(f"Necessidade de caixa P50 = {-np.percentile(res['caixa_min'], 50):.2f} | P90 = {-np.percentile(res['caixa_min'], 10):.2f}")
print(f"Equilíbrio em até 5 anos: {(res['equilibrio'] <= 5).mean():.1%}")
print("H2 suportada" if p_pos - erro >= 0.70 else "H2 não suportada")

# Verificações de robustez
vpl_piso = [s["vp_anos"] + max(s["vp_terminal"], 0) for s in saidas]  # sem perpetuidade de prejuízo
print(f"P(VPL > 0) sem perpetuidade de resultado negativo = {(np.array(vpl_piso) > 0).mean():.1%}")
for corte in (0.9, 1.0):
    sub = res[res["mult_clientes"] >= corte]
    print(f"P(VPL > 0 | ritmo de aquisição >= {corte:.0%} do previsto) = {(sub['vpl'] > 0).mean():.1%} (n = {len(sub)})")
sub = res[(res["mult_clientes"] >= 1.0) & (res["churn"] <= 0.05)]
print(f"P(VPL > 0 | ritmo >= 100% e rotatividade <= 5%) = {(sub['vpl'] > 0).mean():.1%} (n = {len(sub)})")

# 4. H3 - SENSIBILIDADE (correlação de Spearman e tornado)
nomes = {"mult_clientes": "Ritmo de aquisição de clientes", "mult_preco": "Preço praticado",
         "aceite": "Taxa de aceite dos agentes", "custo_var": "Custo variável de IA e voz",
         "mult_fixos": "Custos fixos", "churn": "Rotatividade de clientes"}
spearman = {k: spearmanr(res[k], res["vpl"])[0] for k in nomes}
print("\n=== H3 ===")
for k, v in sorted(spearman.items(), key=lambda x: -abs(x[1])):
    print(f"{nomes[k]:<32} rho = {v:+.3f}")

faixas = {"mult_clientes": (0.50, 1.20), "mult_preco": (0.80, 1.10), "aceite": (0.60, 0.90),
          "custo_var": (0.12, 0.04), "mult_fixos": (1.15, 0.90), "churn": (0.20, 0.00)}
vpl_base = projetar(**BASE)["vpl"]
tornado = []
for k, (ruim, bom) in faixas.items():
    vr = projetar(**{**BASE, k: ruim})["vpl"]
    vb = projetar(**{**BASE, k: bom})["vpl"]
    tornado.append((nomes[k], vr, vb))
tornado.sort(key=lambda x: abs(x[2] - x[1]))
print(f"\nVPL determinístico (base) = {vpl_base:.2f}")
for nome, vr, vb in reversed(tornado):
    print(f"{nome:<32} pior = {vr:+.2f} | melhor = {vb:+.2f} | amplitude = {vb-vr:.2f}")
maior = max(spearman, key=lambda k: abs(spearman[k]))
print("H3 suportada" if maior == "mult_clientes" else "H3 não suportada")

res.to_csv("resultados_monte_carlo.csv", index=False)

# 5. GRÁFICOS
AZUL, LARANJA, TINTA, CINZA = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": CINZA,
                     "xtick.color": CINZA, "ytick.color": CINZA})

fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.linspace(res["vpl"].min(), res["vpl"].max(), 60)
ax.hist(res.loc[res["vpl"] <= 0, "vpl"], bins=bins, color=LARANJA, edgecolor="white", linewidth=0.6,
        label=f"VPL negativo ({1-p_pos:.0%})")
ax.hist(res.loc[res["vpl"] > 0, "vpl"], bins=bins, color=AZUL, edgecolor="white", linewidth=0.6,
        label=f"VPL positivo ({p_pos:.0%})")
ax.axvline(0, color=TINTA, linewidth=1)
ax.axvline(p50, color=CINZA, linewidth=1, linestyle="--")
ax.text(p50 - 0.2, ax.get_ylim()[1] * 0.97, f"mediana R$ {p50:.1f} mi".replace(".", ","), color=CINZA, va="top", ha="right")
ax.set_xlabel("VPL (R$ milhões)")
ax.set_ylabel("Número de cenários")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("grafico_h2_vpl.png", dpi=200)

fig, ax = plt.subplots(figsize=(8, 4.4))
for i, (nome, vr, vb) in enumerate(tornado):
    ax.barh(i, vr - vpl_base, left=vpl_base, color=LARANJA, height=0.6)
    ax.barh(i, vb - vpl_base, left=vpl_base, color=AZUL, height=0.6)
ax.axvline(vpl_base, color=TINTA, linewidth=1)
ax.set_yticks(range(len(tornado)), [t[0] for t in tornado])
ax.set_xlabel("VPL (R$ milhões)")
ax.spines[["top", "right"]].set_visible(False)
from matplotlib.patches import Patch
ax.text(vpl_base, len(tornado) - 0.4, f"cenário-base: R$ {vpl_base:.1f} mi".replace(".", ","), color=CINZA, ha="center", va="bottom")
ax.set_ylim(-0.6, len(tornado) + 0.2)
ax.legend(handles=[Patch(color=LARANJA, label="Premissa no extremo desfavorável"),
                   Patch(color=AZUL, label="Premissa no extremo favorável")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2)
fig.tight_layout()
fig.savefig("grafico_h3_tornado.png", dpi=200)
