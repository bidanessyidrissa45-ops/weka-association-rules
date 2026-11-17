"""Trace les graphes du benchmark Apriori (results/apriori_benchmark.csv).

Usage : python plot.py   (seul matplotlib est nécessaire)
Fonctionne aussi tel quel dans Google Colab après upload du CSV.
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lignes = []
with open("results/apriori_benchmark.csv", newline="") as f:
    for r in csv.DictReader(f, delimiter=";"):
        lignes.append((float(r["support"]), float(r["confiance"]), float(r["temps_ms"]), int(r["nb_regles"])))


def grouper(cle, x):
    """Regroupe les lignes par la colonne `cle` ; renvoie {valeur: (xs, temps, regles)}."""
    g = defaultdict(lambda: ([], [], []))
    for ligne in sorted(lignes, key=lambda l: l[x]):
        xs, ts, ns = g[ligne[cle]]
        xs.append(ligne[x]); ts.append(ligne[2]); ns.append(ligne[3])
    return dict(sorted(g.items()))


par_conf = grouper(1, 0)   # une courbe par confiance, x = support
par_sup = grouper(0, 1)    # une courbe par support, x = confiance

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for conf, (xs, ts, _) in par_conf.items():
    axes[0].plot(xs, ts, marker="o", label=f"conf = {conf:.2f}")
axes[0].set(title="Temps d'exécution vs support", xlabel="support minimal", ylabel="temps (ms)", yscale="log")
axes[0].legend()

for sup, (xs, ts, _) in par_sup.items():
    axes[1].plot(xs, ts, marker="o", label=f"sup = {sup:.2f}")
axes[1].set(title="Temps d'exécution vs confiance", xlabel="confiance minimale", ylabel="temps (ms)", yscale="log")
axes[1].legend(fontsize=7, ncol=2)

for conf, (xs, _, ns) in par_conf.items():
    axes[2].plot(xs, ns, marker="o", label=f"conf = {conf:.2f}")
axes[2].set(title="Nombre de règles vs support", xlabel="support minimal", ylabel="nombre de règles")
axes[2].set_yscale("symlog")
axes[2].legend()

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/apriori_benchmark.png", dpi=150)
print("Graphe enregistré : results/apriori_benchmark.png")
