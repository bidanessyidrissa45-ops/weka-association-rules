# Weka : règles d'association (Apriori) via l'API Java

Projet de Machine Learning réalisé à l'EPITA (Bachelor Cybersécurité, 2026-2027), en groupe.
Objectif : refaire par programme (API Java de Weka 3.8.7) ce qui est fait dans l'interface graphique **Explorer**, puis mesurer les performances d'Apriori.

## Contenu

```
src/WekaAssociation.java   Apriori (4 mesures), prétraitement, Apriori en mode CAR vs J48, validation croisée
src/AprioriBenchmark.java  Temps d'Apriori selon le support et la confiance (sortie CSV)
plot.py                    Graphes du benchmark (matplotlib)
data/                      Jeux de données publics de Weka / UCI : iris, vote, weather.nominal
results/                   Sorties du programme, benchmark CSV et graphe
```

## Ce que fait le programme

| Partie | Travail | Résultat |
|---|---|---|
| Mesures | Apriori avec Confidence, Lift, Leverage et Conviction | règles comparées selon la mesure |
| Prétraitement | Conversion CSV vers ARFF, suppression d'attribut, discrétisation | Apriori refusé sur données brutes (attributs numériques), accepté après discrétisation |
| Iris discrétisé | Apriori en mode CAR (classe en conclusion) comparé à J48 | J48 : 144/150 (96 %), 10 feuilles |
| Épicerie | Apriori à plusieurs supports (0,75 / 0,5 / 0,25 / 0,1) | 0 / 6 / 54 / 283 règles |
| Vote (UCI) | Exploration par parti, Apriori normal et CAR | `physician-fee-freeze=n ==> democrat` (confiance 0,99) |
| Iris brut | J48, validation croisée 10 plis | 144/150 (96 %), 5 feuilles |

## Benchmark

Weka ne démarre pas au support demandé : il part de `1 - delta` et descend par pas de `delta`, et `upperBoundMinSupport` supprime les règles dont le support est trop élevé. Le benchmark fixe donc `delta = 1 - support` et `upperBound = 1`, ce qui donne un seul cycle au support exact (voir `AprioriBenchmark.lancer`).

Résultats : `results/apriori_benchmark.csv` et `results/apriori_benchmark.png`.

## Lancer

1. Installer un JDK 17+ et [Weka 3.8.7](https://ml.cms.waikato.ac.nz/weka/) ; copier `weka.jar` dans `lib/` (non versionné).
2. Compiler et exécuter :

```bash
javac -cp lib/weka.jar -d out src/*.java
java -cp "out:lib/weka.jar" WekaAssociation      # Windows : utiliser ; à la place de :
java -cp "out:lib/weka.jar" AprioriBenchmark
python plot.py
```

Jeu de données non inclus : *Bank Marketing* (`bank-full.csv`, [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing)) à placer dans `data/` pour la partie prétraitement. Sans lui, le benchmark utilise `data/vote.arff`.
