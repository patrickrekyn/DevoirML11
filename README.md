# Rapport de Travaux Pratiques : Introduction au Tuning d'Hyperparamètres
**Module :** Machine Learning  
**Date :** Mercredi 03 mai 2026  
**Auteur :** Rapport Individuel  
**Établissement :** Institut Supérieur Polytechnique de Madagascar (ISPM)  
**Parcours :** ESIIA4 - IGGLIA4 - IMTICIA4 - ISAIA4  

---

## 🎯 Objectif du TP
Comprendre par la pratique comment régler un modèle de Machine Learning (ici, une **Régression Logistique**) afin d'éviter les phénomènes de **sous-apprentissage** (*underfitting*) et de **surapprentissage** (*overfitting*).

---

## 🧠 Zone de Réflexion n°1 : Analyse du Graphique Expérimental

En analysant le graphique représentant l'évolution des précisions (Accuracy) des jeux de *Train* (courbe bleue) et de *Test* (courbe orange) en fonction de l'hyperparamètre $C$ :

### 1. Zone $C \leq 0.001$ : État des scores et comportement du modèle
* **Constat sur les scores :** Les scores sur le jeu d'entraînement (*Train*) et sur le jeu de validation (*Test*) sont tous les deux faibles et stagnent à un niveau sous-optimal.
* **Nom de la situation :** Cette situation s'appelle le **sous-apprentissage** (*underfitting*).
* **Pourquoi le modèle se comporte-t-il ainsi ?** L'hyperparamètre $C$ est l'inverse de la force de régularisation ($C = 1/\lambda$). Lorsque $C$ est très petit, la contrainte de régularisation est extrêmement forte (la "laisse" est très courte). Le modèle est trop bridé, ce qui l'empêche de capturer les relations complexes et les motifs géométriques au sein des variables médicales du jeu de données `breast_cancer`.

### 2. Zone $C \geq 100$ : Écart des courbes et défaut du modèle
* **Constat sur l'écart :** On observe un écart important (un fossé) qui se creuse entre la courbe bleue et la courbe orange. La précision sur le *Train* approche de la perfection (100%), tandis que la précision sur le *Test* stagne ou chute.
* **Comportement (défaut) illustré :** Ce phénomène illustre le **surapprentissage** (*overfitting*). Le modèle dispose d'une liberté totale (laisse trop longue). Au lieu d'apprendre la règle générale de détection des tumeurs, il apprend par cœur le bruit statistique et les spécificités du jeu d'entraînement, ce qui le rend incapable de généraliser sur de nouvelles données.

### 3. Le juste milieu : La valeur optimale de $C$
* Visuellement, le **juste milieu** se situe dans la zone stable où la courbe orange (*Test*) atteint son point culminant (maximum de généralisation) avant que l'écart avec la courbe bleue ne devienne critique. Sur ce graphique à échelle logarithmique, cette valeur optimale se situe généralement entre **$C = 0.1$** et **$C = 1.0$**.

### 4. Le Piège : Évaluation "neutre" et risque de triche (*Data Leakage*)
* **Le jeu de Test est-il encore neutre ?** Non, le jeu de test perd sa neutralité absolue.
* **Risque de triche involontaire (*Data Leakage*) :** Si l'humain ou l'algorithme ajuste l'hyperparamètre $C$ en regardant directement les performances sur le jeu de Test, des informations du jeu de test "fuient" indirectement dans la phase de conception du modèle. Le jeu de test ne fait plus office de données futures inconnues, ce qui donne une estimation biaisée et trop optimiste des performances réelles du modèle en production.

---

## 🛠️ Zone de Réflexion n°2 : La Solution Professionnelle (GridSearchCV)

### 1. Correspondance entre la recherche automatique et l'analyse visuelle
* **Oui**, la valeur optimale de $C$ identifiée automatiquement par `GridSearchCV` (généralement `C = 0.1` ou `C = 1.0` selon le découpage aléatoire) correspond parfaitement à la zone de compromis idéale identifiée visuellement à l'Étape 2. L'algorithme valide scientifiquement l'intuition graphique.

### 2. Pourquoi la Validation Croisée évite le problème du "tâtonnement tricheur"
La validation croisée (*K-Fold Cross-Validation*) résout ce problème grâce à un cloisonnement strict des données :
1. **Verrouillage du Test :** Le jeu de test initial est mis de côté dans un "coffre-fort" hermétique. Il n'est jamais lu pendant la recherche des hyperparamètres.
2. **Découpage interne (ex: $cv=5$) :** Le jeu de *Train* est divisé en 5 blocs. À chaque itération, le modèle s'entraîne sur 4 blocs et s'évalue sur le 5ème bloc restant (appelé bloc de validation).
3. **Évaluation saine :** Les performances utilisées pour élire le meilleur $C$ proviennent uniquement de ces phases de validation internes. Le modèle n'a ainsi jamais "vu" le jeu de test final. L'évaluation finale (`final_score`) reste donc 100% neutre, intègre et représentative.

---

## 🚀 Étape 4 : Extension & Outils de Tuning Avancés

### I. Analyse Multi-paramètres ($C$ + Pénalité)
En combinant l'exploration de la régularisation et du type de norme (`penalty=['l1', 'l2']`) avec le solveur adapté `liblinear`, la recherche automatique permet de découvrir des combinaisons hautement performantes :
* **Pénalité L1 (Lasso) :** Effectue une sélection de variables en annulant purement et simplement les poids ($w$) des variables les moins informatives. C'est idéal pour simplifier le modèle médical.
* **Pénalité L2 (Ridge) :** Réduit proportionnellement la valeur de tous les poids ($w$) sans les annuler, conservant toutes les variables mais limitant leur impact.
* **Résultat :** L'inclusion de la pénalité `L1` offre souvent une précision supérieure ou égale tout en offrant un modèle plus interprétable pour les praticiens de santé.

### II. Comparaison des outils de Tuning Automatique

| Outil / Méthode | Fonctionnement | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **GridSearchCV** *(Scikit-Learn)* | Recherche exhaustive et systématique sur une grille prédéfinie de valeurs. | - Certitude mathématique de trouver la meilleure combinaison de la grille.<br>- Très simple à coder. | - Très lourd et lent.<br>- Souffre de la malédiction de la dimensionnalité si on ajoute beaucoup de paramètres. |
| **RandomizedSearchCV** *(Scikit-Learn)* | Sélectionne et teste un sous-ensemble aléatoire de combinaisons à partir de distributions définies. | - Beaucoup plus rapide.<br>- Permet d'explorer un espace de recherche beaucoup plus large en un temps maîtrisé. | - Ne garantit pas de trouver la combinaison optimale absolue (dépend du hasard). |
| **Optimisation Bayésienne** *(Optuna / Scikit-Optimize)* | Construit un modèle probabiliste des performances passées pour prédire et tester les paramètres les plus prometteurs. | - Ultra-efficace (converge vers l'optimum en très peu d'essais).<br>- Idéal pour le Deep Learning ou les grands modèles. | - Nécessite une bibliothèque externe (`optuna`).<br>- Légèrement plus complexe à paramétrer. |