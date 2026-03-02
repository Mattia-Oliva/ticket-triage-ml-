"""
Dashboard Streamlit per il Triage Automatico dei Ticket con ML.
Tre tab: Classifica Ticket Singolo, Analisi Batch, Metriche e Performance.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Aggiungi la cartella src al path per importare preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import combina_campi, pulisci_testo

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

# ---------------------------------------------------------------------------
# Caricamento modelli (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def carica_modelli():
    """Carica modelli e vectorizer da disco."""
    cat_model = joblib.load(os.path.join(MODELS_DIR, "category_model.pkl"))
    pri_model = joblib.load(os.path.join(MODELS_DIR, "priority_model.pkl"))
    tfidf_cat = joblib.load(os.path.join(MODELS_DIR, "tfidf_category.pkl"))
    tfidf_pri = joblib.load(os.path.join(MODELS_DIR, "tfidf_priority.pkl"))
    return cat_model, pri_model, tfidf_cat, tfidf_pri


def classifica_ticket(title: str, body: str, cat_model, pri_model, tfidf_cat, tfidf_pri):
    """Classifica un singolo ticket per categoria e priorita."""
    testo = pulisci_testo(combina_campi(title, body))
    X_cat = tfidf_cat.transform([testo])
    X_pri = tfidf_pri.transform([testo])

    # Categoria
    cat_pred = cat_model.predict(X_cat)[0]
    if hasattr(cat_model, "predict_proba"):
        cat_proba = cat_model.predict_proba(X_cat)[0]
        cat_classes = cat_model.classes_
    elif hasattr(cat_model, "decision_function"):
        scores = cat_model.decision_function(X_cat)[0]
        exp_scores = np.exp(scores - np.max(scores))
        cat_proba = exp_scores / exp_scores.sum()
        cat_classes = cat_model.classes_
    else:
        cat_proba = None
        cat_classes = None

    # Priorita
    pri_pred = pri_model.predict(X_pri)[0]
    if hasattr(pri_model, "predict_proba"):
        pri_proba = pri_model.predict_proba(X_pri)[0]
        pri_classes = pri_model.classes_
    elif hasattr(pri_model, "decision_function"):
        scores = pri_model.decision_function(X_pri)[0]
        exp_scores = np.exp(scores - np.max(scores))
        pri_proba = exp_scores / exp_scores.sum()
        pri_classes = pri_model.classes_
    else:
        pri_proba = None
        pri_classes = None

    return {
        "categoria": cat_pred,
        "cat_proba": dict(zip(cat_classes, cat_proba)) if cat_proba is not None else {},
        "priorita": pri_pred,
        "pri_proba": dict(zip(pri_classes, pri_proba)) if pri_proba is not None else {},
        "X_cat": X_cat,
        "X_pri": X_pri,
    }


def get_top_features(model, vectorizer, X_vec, predicted_class, top_n=5):
    """Estrae le top N parole piu influenti per la classe predetta."""
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        # LogisticRegression o LinearSVC
        classes = model.classes_
        class_idx = list(classes).index(predicted_class)
        if model.coef_.shape[0] == 1:
            coefs = model.coef_[0]
        else:
            coefs = model.coef_[class_idx]
        # Moltiplica per i valori TF-IDF del documento
        tfidf_values = X_vec.toarray()[0]
        importance = coefs * tfidf_values
        top_indices = np.argsort(importance)[-top_n:][::-1]
        return [(feature_names[i], importance[i]) for i in top_indices if importance[i] > 0]
    elif hasattr(model, "feature_log_prob_"):
        # MultinomialNB
        classes = model.classes_
        class_idx = list(classes).index(predicted_class)
        log_probs = model.feature_log_prob_[class_idx]
        tfidf_values = X_vec.toarray()[0]
        importance = log_probs * tfidf_values
        top_indices = np.argsort(importance)[-top_n:][::-1]
        # Per NB le log-prob sono negative, quindi le piu alte (meno negative)
        # moltiplicate per tfidf danno valori negativi; invertiamo il segno per il display
        return [(feature_names[i], abs(importance[i])) for i in top_indices if tfidf_values[i] > 0]

    return []


# ---------------------------------------------------------------------------
# Colori per badge
# ---------------------------------------------------------------------------

COLORI_CATEGORIA = {
    "Amministrazione": "#2196F3",
    "Tecnico": "#F44336",
    "Commerciale": "#4CAF50",
}

COLORI_PRIORITA = {
    "alta": "#D32F2F",
    "media": "#FF9800",
    "bassa": "#4CAF50",
}


def badge_html(testo: str, colore: str) -> str:
    """Genera un badge colorato in HTML."""
    return (
        f'<span style="background-color:{colore};color:white;padding:6px 16px;'
        f'border-radius:20px;font-weight:bold;font-size:1.1em;">{testo}</span>'
    )


# ---------------------------------------------------------------------------
# App principale
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Triage Ticket ML",
        page_icon="🎫",
        layout="wide",
    )

    st.title("Triage Automatico Ticket con ML")
    st.caption("Classificazione automatica di ticket aziendali per categoria e priorita")

    cat_model, pri_model, tfidf_cat, tfidf_pri = carica_modelli()

    tab1, tab2, tab3 = st.tabs([
        "Classifica Ticket Singolo",
        "Analisi Batch",
        "Metriche e Performance",
    ])

    # -----------------------------------------------------------------------
    # Tab 1: Classifica Ticket Singolo
    # -----------------------------------------------------------------------
    with tab1:
        st.header("Classifica un Ticket")

        col_input, col_output = st.columns([1, 1])

        with col_input:
            oggetto = st.text_input("Oggetto del ticket", placeholder="Es: Problema con la stampante di rete")
            descrizione = st.text_area(
                "Descrizione del ticket",
                height=150,
                placeholder="Es: La stampante del secondo piano non risponde ai comandi di stampa da questa mattina...",
            )
            btn_classifica = st.button("Classifica", type="primary", use_container_width=True)

        with col_output:
            if btn_classifica:
                if not oggetto and not descrizione:
                    st.warning("Inserisci almeno l'oggetto o la descrizione del ticket.")
                else:
                    result = classifica_ticket(
                        oggetto, descrizione, cat_model, pri_model, tfidf_cat, tfidf_pri
                    )

                    st.subheader("Risultato")

                    # Categoria
                    cat = result["categoria"]
                    cat_color = COLORI_CATEGORIA.get(cat, "#607D8B")
                    cat_prob = result["cat_proba"].get(cat, 0) * 100
                    st.markdown(
                        f"**Categoria:** {badge_html(cat, cat_color)} &nbsp; "
                        f"<span style='color:grey;'>({cat_prob:.1f}%)</span>",
                        unsafe_allow_html=True,
                    )

                    # Probabilita categoria
                    if result["cat_proba"]:
                        st.markdown("*Probabilita per categoria:*")
                        for cls in sorted(result["cat_proba"].keys()):
                            prob = result["cat_proba"][cls] * 100
                            st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")

                    st.divider()

                    # Priorita
                    pri = result["priorita"]
                    pri_color = COLORI_PRIORITA.get(pri, "#607D8B")
                    pri_prob = result["pri_proba"].get(pri, 0) * 100
                    st.markdown(
                        f"**Priorita:** {badge_html(pri.upper(), pri_color)} &nbsp; "
                        f"<span style='color:grey;'>({pri_prob:.1f}%)</span>",
                        unsafe_allow_html=True,
                    )

                    # Probabilita priorita
                    if result["pri_proba"]:
                        st.markdown("*Probabilita per priorita:*")
                        for cls in ["alta", "media", "bassa"]:
                            if cls in result["pri_proba"]:
                                prob = result["pri_proba"][cls] * 100
                                st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")

                    st.divider()

                    # Top features
                    st.markdown("**Top 5 parole piu influenti:**")
                    top_cat = get_top_features(
                        cat_model, tfidf_cat, result["X_cat"], cat, top_n=5
                    )
                    top_pri = get_top_features(
                        pri_model, tfidf_pri, result["X_pri"], pri, top_n=5
                    )

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown(f"*Per categoria ({cat}):*")
                        for word, score in top_cat:
                            st.markdown(f"- **{word}** ({score:.3f})")
                    with col_f2:
                        st.markdown(f"*Per priorita ({pri}):*")
                        for word, score in top_pri:
                            st.markdown(f"- **{word}** ({score:.3f})")

    # -----------------------------------------------------------------------
    # Tab 2: Analisi Batch
    # -----------------------------------------------------------------------
    with tab2:
        st.header("Analisi Batch")
        st.markdown("Carica un file CSV con colonne `title` e `body` per classificare piu ticket.")

        uploaded_file = st.file_uploader("Carica file CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Errore nella lettura del file: {e}")
                df = None

            if df is not None:
                # Verifica colonne
                required_cols = {"title", "body"}
                if not required_cols.issubset(set(df.columns)):
                    st.error(f"Il file deve contenere le colonne: {required_cols}. Trovate: {set(df.columns)}")
                else:
                    st.subheader("Anteprima dati caricati")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"Totale righe: {len(df)}")

                    if st.button("Predici Batch", type="primary", use_container_width=True):
                        with st.spinner("Classificazione in corso..."):
                            risultati = []
                            for _, row in df.iterrows():
                                res = classifica_ticket(
                                    str(row["title"]), str(row["body"]),
                                    cat_model, pri_model, tfidf_cat, tfidf_pri,
                                )
                                risultati.append({
                                    "title": row["title"],
                                    "body": row["body"],
                                    "categoria_prevista": res["categoria"],
                                    "priorita_prevista": res["priorita"],
                                })

                            df_result = pd.DataFrame(risultati)

                        st.subheader("Risultati Classificazione")
                        st.dataframe(df_result, use_container_width=True)

                        # Statistiche rapide
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Distribuzione Categorie:**")
                            st.dataframe(
                                df_result["categoria_prevista"].value_counts().reset_index(),
                                use_container_width=True,
                            )
                        with col2:
                            st.markdown("**Distribuzione Priorita:**")
                            st.dataframe(
                                df_result["priorita_prevista"].value_counts().reset_index(),
                                use_container_width=True,
                            )

                        # Download
                        csv_data = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Scarica CSV Risultati",
                            data=csv_data,
                            file_name="ticket_classificati.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

    # -----------------------------------------------------------------------
    # Tab 3: Metriche e Performance
    # -----------------------------------------------------------------------
    with tab3:
        st.header("Metriche e Performance dei Modelli")

        # Tabella riassuntiva
        report_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()

            st.subheader("Report di Classificazione")
            st.code(report_text, language=None)

        st.divider()

        # Grafici
        st.subheader("Confusion Matrix")
        col_cm1, col_cm2 = st.columns(2)

        cm_cat_path = os.path.join(OUTPUTS_DIR, "confusion_matrix_categoria.png")
        cm_pri_path = os.path.join(OUTPUTS_DIR, "confusion_matrix_priorita.png")

        with col_cm1:
            if os.path.exists(cm_cat_path):
                st.image(cm_cat_path, caption="Confusion Matrix - Categoria", use_container_width=True)
        with col_cm2:
            if os.path.exists(cm_pri_path):
                st.image(cm_pri_path, caption="Confusion Matrix - Priorita", use_container_width=True)

        st.divider()

        col_g1, col_g2 = st.columns(2)

        f1_path = os.path.join(OUTPUTS_DIR, "f1_per_classe.png")
        confronto_path = os.path.join(OUTPUTS_DIR, "confronto_modelli.png")

        with col_g1:
            st.subheader("F1-Score per Classe")
            if os.path.exists(f1_path):
                st.image(f1_path, caption="F1-score per Classe", use_container_width=True)

        with col_g2:
            st.subheader("Confronto Modelli")
            if os.path.exists(confronto_path):
                st.image(confronto_path, caption="Confronto Accuracy Modelli", use_container_width=True)

        # Predizioni test set
        st.divider()
        pred_path = os.path.join(OUTPUTS_DIR, "predictions_test.csv")
        if os.path.exists(pred_path):
            st.subheader("Predizioni sul Test Set")
            df_pred = pd.read_csv(pred_path)
            st.dataframe(df_pred, use_container_width=True)
            st.caption(f"Totale: {len(df_pred)} ticket nel test set")


if __name__ == "__main__":
    main()
