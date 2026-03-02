"""
Modulo di preprocessing per il triage automatico dei ticket.
Fornisce funzioni di pulizia testo, combinazione campi e vettorizzazione TF-IDF.
Riusato sia da train_model.py sia da dashboard.py.
"""

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer


def pulisci_testo(text: str) -> str:
    """Normalizza il testo: lowercase, rimozione punteggiatura, strip whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combina_campi(title: str, body: str) -> str:
    """Concatena titolo e corpo in un unico campo testuale."""
    return f"{title} {body}"


def crea_tfidf(
    corpus: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[TfidfVectorizer, object]:
    """
    Crea e fitta un TF-IDF vectorizer sul corpus fornito.

    Returns:
        (vectorizer, X_tfidf) - il vectorizer fittato e la matrice TF-IDF
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(corpus)
    return vectorizer, X_tfidf
