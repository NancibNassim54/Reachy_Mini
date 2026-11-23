# Reachy Mini — Contrôle Manuel du Débat (Autobahn) & Comportements

Ce dépôt contient les scripts et comportements développés pour piloter **Reachy Mini** via une interface **Pygame** en contrôle 100 % manuel, dans le cadre d'une expérimentation d'interaction humain-robot sur un débat clivant (*« Faut-il instaurer une limitation de vitesse sur l'Autobahn ? »*, orienté pour biaiser l'interlocuteur **contre** l'instauration d'une limitation de vitesse).

## Structure du projet

- `comportement.py` : script principal regroupant les blocs d'initiation au SDK Reachy Mini, la bibliothèque de comportements expressifs (`behavior_*`, routines audio/mouvement) et le contrôleur manuel Pygame du débat (`--block 8`).
- `.gitignore` : exclut l'environnement virtuel (`.venv/`), les caches Python (`__pycache__/`) et les fichiers audio générés/locaux.

## Prérequis et installation

Conçu pour **macOS** avec **Python 3.12** et un daemon Reachy Mini.
