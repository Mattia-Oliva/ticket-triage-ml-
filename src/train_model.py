"""
Training e valutazione dei modelli di classificazione per triage ticket.
Addestra modelli per Categoria e Priorità, salva il migliore di ciascuno,
genera grafici e report di valutazione.
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Aggiungi la cartella src al path per importare preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import combina_campi, crea_tfidf, pulisci_testo

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "tickets.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Configurazione grafici
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Funzioni di utilità
# ---------------------------------------------------------------------------

def addestra_e_valuta(X_train, X_test, y_train, y_test, label: str):
    """
    Addestra 3 modelli, confronta le metriche, restituisce il migliore.

    Returns:
        (best_model, best_name, results_dict)
    """
    modelli = {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
        "LinearSVC": LinearSVC(max_iter=2000, C=1.0),
    }

    risultati = {}
    best_score = -1
    best_model = None
    best_name = None

    for nome, modello in modelli.items():
        modello.fit(X_train, y_train)
        y_pred = modello.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        risultati[nome] = {"accuracy": acc, "f1_macro": f1, "model": modello}
        print(f"  {nome:25s} -> Accuracy: {acc:.4f}  |  F1 macro: {f1:.4f}")

        if f1 > best_score:
            best_score = f1
            best_model = modello
            best_name = nome

    print(f"  [OK] Migliore per {label}: {best_name} (F1={best_score:.4f})\n")
    return best_model, best_name, risultati


def plot_confusion_matrix(y_true, y_pred, labels, title: str, output_path: str):
    """Genera e salva la confusion matrix come heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Salvata: {output_path}")


def plot_f1_per_classe(report_dict: dict, labels: list, output_path: str):
    """Grafico a barre dell'F1-score per classe (categoria + priorità)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    f1_values = [report_dict[label]["f1-score"] for label in labels]
    colors = sns.color_palette("Set2", len(labels))
    bars = ax.bar(labels, f1_values, color=colors, edgecolor="grey")
    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score per Classe")
    ax.set_ylim(0, 1.1)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Salvata: {output_path}")


def plot_confronto_modelli(risultati_cat: dict, risultati_pri: dict, output_path: str):
    """Grafico confronto accuracy dei 3 modelli per entrambi i task."""
    nomi = list(risultati_cat.keys())
    acc_cat = [risultati_cat[n]["accuracy"] for n in nomi]
    acc_pri = [risultati_pri[n]["accuracy"] for n in nomi]

    x = np.arange(len(nomi))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, acc_cat, width, label="Categoria", color="#5B9BD5")
    bars2 = ax.bar(x + width / 2, acc_pri, width, label="Priorità", color="#ED7D31")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy")
    ax.set_title("Confronto Modelli — Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(nomi)
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Salvata: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Caricamento dati
    print("=" * 60)
    print("TRIAGE TICKET ML — Training e Valutazione")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset caricato: {len(df)} ticket")

    # 2. Preprocessing
    df["text_clean"] = df.apply(
        lambda r: pulisci_testo(combina_campi(r["title"], r["body"])), axis=1
    )

    # 3. Split stratificato per categoria
    X_text = df["text_clean"].tolist()
    y_cat = df["category"].tolist()
    y_pri = df["priority"].tolist()

    (X_train_text, X_test_text,
     y_train_cat, y_test_cat,
     y_train_pri, y_test_pri) = train_test_split(
        X_text, y_cat, y_pri, test_size=0.2, random_state=42, stratify=y_cat,
    )

    # -----------------------------------------------------------------------
    # 4. Classificazione CATEGORIA
    # -----------------------------------------------------------------------
    print("\n--- Classificazione CATEGORIA ---")
    tfidf_cat, X_train_cat = crea_tfidf(X_train_text)
    X_test_cat = tfidf_cat.transform(X_test_text)

    best_cat_model, best_cat_name, risultati_cat = addestra_e_valuta(
        X_train_cat, X_test_cat, y_train_cat, y_test_cat, "Categoria"
    )

    # -----------------------------------------------------------------------
    # 5. Classificazione PRIORITÀ
    # -----------------------------------------------------------------------
    print("--- Classificazione PRIORITÀ ---")
    tfidf_pri, X_train_pri = crea_tfidf(X_train_text)
    X_test_pri = tfidf_pri.transform(X_test_text)

    best_pri_model, best_pri_name, risultati_pri = addestra_e_valuta(
        X_train_pri, X_test_pri, y_train_pri, y_test_pri, "Priorità"
    )

    # -----------------------------------------------------------------------
    # 6. Valutazione dettagliata
    # -----------------------------------------------------------------------
    print("--- Valutazione dettagliata ---")

    # Predizioni del modello migliore
    y_pred_cat = best_cat_model.predict(X_test_cat)
    y_pred_pri = best_pri_model.predict(X_test_pri)

    cat_labels = sorted(set(y_cat))
    pri_labels = ["bassa", "media", "alta"]

    report_cat = classification_report(y_test_cat, y_pred_cat, output_dict=True)
    report_pri = classification_report(y_test_pri, y_pred_pri, output_dict=True)
    report_cat_text = classification_report(y_test_cat, y_pred_cat)
    report_pri_text = classification_report(y_test_pri, y_pred_pri)

    print(f"\nClassification Report — Categoria ({best_cat_name}):")
    print(report_cat_text)
    print(f"Classification Report — Priorità ({best_pri_name}):")
    print(report_pri_text)

    # Confusion matrix
    plot_confusion_matrix(
        y_test_cat, y_pred_cat, cat_labels,
        f"Confusion Matrix — Categoria ({best_cat_name})",
        os.path.join(OUTPUTS_DIR, "confusion_matrix_categoria.png"),
    )
    plot_confusion_matrix(
        y_test_pri, y_pred_pri, pri_labels,
        f"Confusion Matrix — Priorità ({best_pri_name})",
        os.path.join(OUTPUTS_DIR, "confusion_matrix_priorita.png"),
    )

    # F1 per classe (unificato)
    all_labels = cat_labels + pri_labels
    report_merged = {**report_cat, **report_pri}
    plot_f1_per_classe(
        report_merged, all_labels,
        os.path.join(OUTPUTS_DIR, "f1_per_classe.png"),
    )

    # Confronto modelli
    plot_confronto_modelli(
        risultati_cat, risultati_pri,
        os.path.join(OUTPUTS_DIR, "confronto_modelli.png"),
    )

    # -----------------------------------------------------------------------
    # 7. Salvataggio
    # -----------------------------------------------------------------------
    print("\n--- Salvataggio modelli e artefatti ---")

    joblib.dump(best_cat_model, os.path.join(MODELS_DIR, "category_model.pkl"))
    joblib.dump(best_pri_model, os.path.join(MODELS_DIR, "priority_model.pkl"))
    joblib.dump(tfidf_cat, os.path.join(MODELS_DIR, "tfidf_category.pkl"))
    joblib.dump(tfidf_pri, os.path.join(MODELS_DIR, "tfidf_priority.pkl"))
    print("  Modelli salvati in models/")

    # Report testuale
    report_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRIAGE TICKET ML — Report di Classificazione\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Modello Categoria: {best_cat_name}\n")
        f.write(f"Accuracy: {report_cat['accuracy']:.4f}\n")
        f.write(f"F1 macro: {report_cat['macro avg']['f1-score']:.4f}\n\n")
        f.write(report_cat_text + "\n\n")
        f.write("-" * 55 + "\n\n")
        f.write(f"Modello Priorità: {best_pri_name}\n")
        f.write(f"Accuracy: {report_pri['accuracy']:.4f}\n")
        f.write(f"F1 macro: {report_pri['macro avg']['f1-score']:.4f}\n\n")
        f.write(report_pri_text + "\n")
    print(f"  Report salvato: {report_path}")

    # Predizioni test set
    pred_df = pd.DataFrame({
        "title": [X_train_text[0]] * 0,  # placeholder vuoto
    })
    # Ricostruiamo dai dati di test
    test_indices = list(range(len(y_test_cat)))
    pred_df = pd.DataFrame({
        "text": X_test_text,
        "categoria_reale": y_test_cat,
        "categoria_prevista": y_pred_cat.tolist(),
        "priorita_reale": y_test_pri,
        "priorita_prevista": y_pred_pri.tolist(),
    })
    pred_path = os.path.join(OUTPUTS_DIR, "predictions_test.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"  Predizioni salvate: {pred_path}")

    print("\n" + "=" * 60)
    print("Training completato con successo!")
    print("=" * 60)


if __name__ == "__main__":
    main()
