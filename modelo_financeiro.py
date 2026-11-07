"""Modelo financeiro determinístico da plataforma (base das hipóteses H2 e H3).

Reproduz a projeção de cinco anos da seção 6.2 da versão acadêmica.
Valores em R$ milhões.
"""
import numpy as np

ANOS = 5
NOVOS_CLIENTES = np.array([6, 8, 8, 10, 10], dtype=float)      # novos clientes por ano
ASSINATURA = np.array([0.180, 0.220, 0.230, 0.240, 0.250])     # R$ mi por cliente por ano
NOVOS_AGENTES_BASE = np.array([0, 4, 8, 12, 16], dtype=float)  # agentes extras em clientes da base
DIAGNOSTICO = 0.020                                            # R$ mi por novo cliente
AGENTE = 0.050                                                 # R$ mi por agente, pago após aceite
DESCONTO_ANO1 = 0.75                                           # desconto aos clientes de design
CUSTOS_FIXOS = np.array([2.2, 3.4, 4.3, 5.4, 6.4])
TRIBUTOS = 0.14
TAXA_DESCONTO = 0.25
CRESCIMENTO_PERPETUIDADE = 0.05

BASE = dict(mult_clientes=1.0, mult_preco=1.0, aceite=0.80, custo_var=0.06,
            mult_fixos=1.0, churn=0.0)


def projetar(mult_clientes=1.0, mult_preco=1.0, aceite=0.80, custo_var=0.06,
             mult_fixos=1.0, churn=0.0):
    """Retorna dicionário com séries anuais, VPL e necessidade máxima de caixa."""
    novos = NOVOS_CLIENTES * mult_clientes
    fim = np.zeros(ANOS)
    anterior = 0.0
    for t in range(ANOS):
        fim[t] = anterior * (1 - churn) + novos[t]
        anterior = fim[t]
    inicio = np.concatenate([[0.0], fim[:-1]])
    medio = (inicio + fim) / 2                       # clientes médios no ano
    medio[0] = fim[0] / 4                            # ano 1: clientes entram no 2º semestre (validação)

    assinaturas = medio * ASSINATURA * mult_preco
    projeto_novo = (DIAGNOSTICO + AGENTE * aceite) * mult_preco
    fator_ano = np.array([DESCONTO_ANO1, 1, 1, 1, 1])
    projetos = (novos * projeto_novo * fator_ano
                + NOVOS_AGENTES_BASE * mult_clientes * AGENTE * aceite * mult_preco)
    receita = assinaturas + projetos
    resultado = receita * (1 - TRIBUTOS - custo_var) - CUSTOS_FIXOS * mult_fixos

    fator = (1 + TAXA_DESCONTO) ** np.arange(1, ANOS + 1)
    vp_anos = np.sum(resultado / fator)
    terminal = resultado[-1] * (1 + CRESCIMENTO_PERPETUIDADE) / (TAXA_DESCONTO - CRESCIMENTO_PERPETUIDADE)
    vpl = vp_anos + terminal / fator[-1]
    caixa_min = np.min(np.cumsum(resultado))
    equilibrio = next((t + 1 for t in range(ANOS) if resultado[t] > 0), None)
    return dict(assinaturas=assinaturas, projetos=projetos, receita=receita,
                resultado=resultado, vp_anos=vp_anos, vp_terminal=terminal / fator[-1],
                vpl=vpl, caixa_min=caixa_min, ano_equilibrio=equilibrio)


if __name__ == "__main__":
    r = projetar(**BASE)
    np.set_printoptions(precision=2, suppress=True)
    print("Receita de assinaturas:", r["assinaturas"])
    print("Receita de projetos:   ", r["projetos"])
    print("Receita bruta:         ", r["receita"])
    print("Resultado operacional: ", r["resultado"])
    print(f"VP dos 5 anos: {r['vp_anos']:.2f} | VP terminal: {r['vp_terminal']:.2f} | VPL: {r['vpl']:.2f}")
    print(f"Necessidade máxima de caixa: {r['caixa_min']:.2f} | Ano de equilíbrio: {r['ano_equilibrio']}")
