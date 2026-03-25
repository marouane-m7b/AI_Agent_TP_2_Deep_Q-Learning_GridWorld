# TP 2 — Agent Double Deep Q-Network (Double DQN)

**Module :** SMA et IAD  
**Filière :** Master SDIA  
**Professeur :** Prof. RETAL SARA  

---

## Description du projet

Ce projet implémente un agent **Double Deep Q-Network (Double DQN)** capable d'apprendre à naviguer dans un environnement **GridWorld 4×4**. L'agent part du coin supérieur gauche (0,0) et doit atteindre l'objectif en (3,3), tout en évitant un obstacle en (1,1).

---

## Environnement GridWorld

```
(0,0) START  →  (0,1)  →  (0,2)  →  (0,3)
  ↓               ↓               ↓               ↓
(1,0)       →  (1,1) ❌  →  (1,2)  →  (1,3)
  ↓               ↓               ↓               ↓
(2,0)       →  (2,1)  →  (2,2)  →  (2,3)
  ↓               ↓               ↓               ↓
(3,0)       →  (3,1)  →  (3,2)  →  (3,3) 🏆 GOAL
```

| Événement        | Récompense |
|------------------|------------|
| Atteindre l'objectif (3,3) | +10 |
| Tomber sur l'obstacle (1,1) | -5 |
| Déplacement normal | -1 |

---

## Différence entre DQN et Double DQN

### DQN classique — le problème

Dans le DQN standard, **un seul réseau** est utilisé pour deux tâches à la fois :
- **Sélectionner** la meilleure action
- **Évaluer** la valeur de cette action

Cela crée un biais de **surestimation** des valeurs Q, ce qui rend l'apprentissage instable.

```
target = r + γ · max Q(s', a')   ← même réseau pour tout → biais
```

### Double DQN — la solution

Le Double DQN utilise **deux réseaux distincts** :

| Réseau | Rôle | Mise à jour |
|--------|------|-------------|
| Réseau principal (online) | Sélectionne la meilleure action | À chaque étape |
| Réseau cible (target) | Évalue la valeur de cette action | Tous les 10 épisodes |

```
a*     = argmax Q_online(s', a)       ← online  SÉLECTIONNE
target = r + γ · Q_target(s', a*)    ← target  ÉVALUE
```

Cette séparation **élimine le biais de surestimation** et stabilise l'apprentissage.

---

## Architecture du réseau de neurones

```
Entrée : vecteur de 16 valeurs (grille 4×4 aplatie)
         ↓
Couche cachée 1 : 24 neurones, activation ReLU
         ↓
Couche cachée 2 : 24 neurones, activation ReLU
         ↓
Sortie : 4 valeurs Q (une par action : haut, bas, gauche, droite)
```

---

## Paramètres d'entraînement

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `EPISODES` | 1000 | Nombre total d'épisodes |
| `GAMMA` | 0.9 | Facteur de réduction (importance du futur) |
| `LEARNING_RATE` | 0.01 | Taux d'apprentissage |
| `EPSILON` | 1.0 → 0.01 | Exploration initiale → minimale |
| `EPSILON_DECAY` | 0.995 | Décroissance exponentielle de l'exploration |
| `BATCH_SIZE` | 32 | Taille du batch pour le replay |
| `MEMORY_SIZE` | 2000 | Capacité de la mémoire d'expériences |
| `UPDATE_TARGET_EVERY` | 10 | Synchronisation du réseau cible (épisodes) |

---

## Stratégie ε-greedy

Au début, l'agent **explore** beaucoup (ε = 1.0 → actions aléatoires).  
Au fur et à mesure, il **exploite** de plus en plus ses connaissances (ε → 0.01).

```
ε = 1.0   →   actions aléatoires (exploration totale)
ε = 0.5   →   50% aléatoire, 50% réseau
ε = 0.01  →   99% réseau, 1% aléatoire (exploitation)
```

---

## Résultats d'entraînement

| Phase | Épisodes | Score moyen | Comportement |
|-------|----------|-------------|--------------|
| Exploration | 1 – 200 | -50 à -80 | Actions aléatoires |
| Apprentissage | 200 – 500 | -20 à 0 | Début de la convergence |
| Convergence | 500 – 1000 | +5.0 | Chemin optimal trouvé |

**Score final : +5.0 constant**  
Explication : le chemin le plus court = 6 pas × (-1) + (+10) = **+4**, mais l'agent atteint régulièrement **+5**, ce qui correspond au chemin optimal.

---

## Résultats de test (après entraînement)

Test sur 20 épisodes avec le modèle sauvegardé (ε = 0, pas d'exploration) :

```
Récompense moyenne :  5.00
Meilleure récompense : 5.00
Pire récompense :      5.00
Taux de succès :       100.0%
```

L'agent atteint l'objectif **à chaque fois**, en prenant toujours le chemin le plus court.

---

## Structure des fichiers

```
Devoir/
├── doubleDqn.py              # Code principal — agent Double DQN
├── script.py                 # Script de test du modèle entraîné
├── double_dqn_model.keras    # Modèle sauvegardé après entraînement
└── README.md                 # Ce fichier
```

---

## Lancer le projet

**Prérequis :**
```bash
pip install tensorflow numpy
```

**Entraînement :**
```bash
python doubleDqn.py
```

**Test du modèle :**
```bash
python script.py
```

---

## Conclusion

L'implémentation du **Double DQN** améliore significativement la stabilité de l'apprentissage par rapport au DQN classique. En séparant la sélection et l'évaluation des actions entre deux réseaux distincts, on élimine le biais de surestimation et on obtient une convergence plus rapide et plus fiable. Les résultats obtenus — **100% de succès avec un score optimal de 5.0** — confirment l'efficacité de cette approche.