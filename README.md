# Ticket Triage ML

Sistema di **triage automatico dei ticket aziendali** basato su Machine Learning.
Classifica i ticket per **categoria** (Amministrazione / Tecnico / Commerciale) e **priorita** (bassa / media / alta) a partire dal testo, con una dashboard interattiva per l'uso quotidiano.

---

## Come funziona

Un ticket arriva con un oggetto e una descrizione testuale. Il sistema:

1. Pulisce e normalizza il testo
2. Lo trasforma in un vettore numerico tramite **TF-IDF** (unigrammi + bigrammi)
3. Lo classifica con due modelli indipendenti: uno per la categoria, uno per la priorita
4. Restituisce la predizione con le probabilita per classe e le **parole chiave** che hanno guidato la decisione

Il tutto in millisecondi, senza GPU.

---

## Quick Start

```bash
# 1. Clona il repo
git clone https://github.com/Mattia-Oliva/ticket-triage-ml.git
cd ticket-triage-ml

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Genera il dataset sintetico (500 ticket)
python src/genera_dataset.py

# 4. Addestra i modelli
python src/train_model.py

# 5. Avvia la dashboard
streamlit run src/dashboard.py
```

La pipeline completa (passi 3-4) gira in meno di un minuto su qualsiasi PC.

---

## Dataset

Il file `data/tickets.csv` contiene **500 ticket sintetici** generati con un sistema di template combinatori. Ogni ticket ha:

| Campo | Descrizione |
|---|---|
| `id` | Identificativo univoco |
| `title` | Oggetto del ticket |
| `body` | Descrizione estesa |
| `category` | Amministrazione, Tecnico o Commerciale |
| `priority` | bassa, media o alta |

La distribuzione e bilanciata per categoria (~167 ciascuna) con priorita realistiche (40% bassa, 35% media, 25% alta). Il seed e fisso (`random.seed(42)`) per garantire riproducibilita.

---

## Modelli

Per ciascun task vengono addestrati e confrontati **3 classificatori**:

| Algoritmo | Tipo | Categoria | Priorita |
|---|---|---|---|
| MultinomialNB | Probabilistico | Acc 1.00 | Acc 0.81 |
| LogisticRegression | Discriminativo | Acc 1.00 | Acc 0.96 |
| LinearSVC | Margine massimo | Acc 1.00 | Acc 1.00 |

Il sistema seleziona automaticamente il migliore per F1-score macro:
- **Categoria** &rarr; MultinomialNB (tutti a 1.00, si sceglie il piu semplice)
- **Priorita** &rarr; LinearSVC (unico a raggiungere il 100%)

I modelli serializzati occupano meno di **250 KB** in totale.

---

## Dashboard

Tre tab per tre casi d'uso diversi:

### Classifica Ticket Singolo
Inserisci oggetto e descrizione, ottieni:
- Categoria e priorita previste con badge colorati
- Probabilita per ogni classe (barre di progresso)
- Top 5 parole piu influenti nella decisione

### Analisi Batch
Carica un CSV con colonne `title` e `body`:
- Preview dei dati
- Tabella risultati con categoria e priorita previste
- Download CSV dei risultati

### Metriche e Performance
- Confusion matrix per entrambi i task
- Grafico F1-score per classe
- Confronto accuracy tra i 3 modelli
- Classification report completo

---

## Struttura del Progetto

```
ticket-triage-ml/
├── data/
│   └── tickets.csv                    # Dataset (500 ticket)
├── src/
│   ├── genera_dataset.py              # Generatore dataset sintetico
│   ├── preprocessing.py               # Pulizia testo + TF-IDF
│   ├── train_model.py                 # Training + valutazione + salvataggio
│   └── dashboard.py                   # Dashboard Streamlit
├── models/
│   ├── category_model.pkl             # Modello categoria (MultinomialNB)
│   ├── priority_model.pkl             # Modello priorita (LinearSVC)
│   ├── tfidf_category.pkl             # Vectorizer TF-IDF categoria
│   └── tfidf_priority.pkl             # Vectorizer TF-IDF priorita
├── outputs/
│   ├── confusion_matrix_categoria.png
│   ├── confusion_matrix_priorita.png
│   ├── f1_per_classe.png
│   ├── confronto_modelli.png
│   ├── classification_report.txt
│   └── predictions_test.csv
├── requirements.txt
└── README.md
```

---

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Linguaggio | Python 3 |
| Machine Learning | scikit-learn |
| Feature Extraction | TF-IDF (max 5000 feature, bigrammi, TF sublineare) |
| Dashboard | Streamlit |
| Grafici | matplotlib + seaborn |
| Dati | pandas + numpy |
| Serializzazione | joblib |

---

## Pipeline in dettaglio

```
tickets.csv
    |
    v
[preprocessing.py]  -->  lowercase, rimuovi punteggiatura, concatena titolo+corpo
    |
    v
[TF-IDF Vectorizer]  -->  unigrammi + bigrammi, max 5000 feature
    |
    v
[train_model.py]  -->  3 modelli x 2 task, split 80/20 stratificato
    |
    v
Modello migliore per task  -->  serializzato in models/*.pkl
    |
    v
[dashboard.py]  -->  carica modelli, classifica in tempo reale
```

---

## Outputs generati

Dopo il training (`python src/train_model.py`), nella cartella `outputs/` trovi:

| File | Contenuto |
|---|---|
| `confusion_matrix_categoria.png` | Heatmap errori classificazione categoria |
| `confusion_matrix_priorita.png` | Heatmap errori classificazione priorita |
| `f1_per_classe.png` | F1-score per ognuna delle 6 classi |
| `confronto_modelli.png` | Accuracy dei 3 modelli a confronto |
| `classification_report.txt` | Report precision/recall/F1 dettagliato |
| `predictions_test.csv` | Predizioni su ogni ticket del test set |

---

## Licenza

Progetto accademico — Project Work per il CdS in Informatica per le Aziende Digitali (L-31), Universita Telematica Pegaso.
