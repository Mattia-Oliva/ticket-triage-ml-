"""
Generatore di dataset sintetico per il triage automatico di ticket aziendali.
Produce 500 ticket bilanciati per categoria (Amministrazione, Tecnico, Commerciale)
con priorità assegnata in base a parole chiave di urgenza.
"""

import csv
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Template per categoria
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Amministrazione": {
        "soggetti": [
            "fattura", "pagamento", "rimborso", "F24", "bonifico",
            "nota di credito", "scadenza fiscale", "bilancio", "partita IVA",
            "dichiarazione IVA", "contributo INPS", "ricevuta", "estratto conto",
            "registro contabile", "ravvedimento operoso",
        ],
        "verbi": [
            "non è stato registrato", "risulta errato", "è in sospeso",
            "deve essere corretto", "non è stato emesso",
            "è stato rifiutato", "va ricalcolato", "presenta anomalie",
            "necessita di verifica", "non corrisponde",
        ],
        "titoli": [
            "Problema con {soggetto}",
            "Richiesta correzione {soggetto}",
            "Segnalazione errore su {soggetto}",
            "Verifica {soggetto}",
            "{soggetto} non conforme",
            "Anomalia {soggetto}",
        ],
        "frasi_corpo": [
            "Il/la {soggetto} relativo al mese corrente {verbo}.",
            "Si richiede la rettifica del/della {soggetto} in quanto {verbo}.",
            "Segnalo che il/la {soggetto} {verbo} e necessita intervento.",
            "Buongiorno, il/la {soggetto} {verbo}. Chiedo assistenza.",
            "Il documento {soggetto} {verbo}, si prega di verificare.",
            "In riferimento al/alla {soggetto}: {verbo}. Attendo riscontro.",
        ],
    },
    "Tecnico": {
        "soggetti": [
            "errore di sistema", "crash applicazione", "bug nel software",
            "aggiornamento firmware", "server aziendale", "rete interna",
            "stampante di rete", "software gestionale", "password utente",
            "lentezza del PC", "schermata blu", "connessione VPN",
            "database", "backup", "firewall",
        ],
        "verbi": [
            "non funziona correttamente", "genera un errore",
            "si blocca all'avvio", "è irraggiungibile",
            "non risponde ai comandi", "mostra messaggi di errore",
            "ha smesso di funzionare", "presenta malfunzionamenti",
            "richiede aggiornamento", "non si avvia",
        ],
        "titoli": [
            "Problema con {soggetto}",
            "Malfunzionamento {soggetto}",
            "Errore {soggetto}",
            "Assistenza per {soggetto}",
            "{soggetto} non disponibile",
            "Guasto {soggetto}",
        ],
        "frasi_corpo": [
            "Il/la {soggetto} {verbo} da questa mattina.",
            "Da ieri il/la {soggetto} {verbo}. Impossibile lavorare.",
            "Segnalo che {soggetto} {verbo}. Serve intervento tecnico.",
            "Ho riscontrato che il/la {soggetto} {verbo}.",
            "Il/la {soggetto} {verbo}. Ho già provato a riavviare.",
            "Urgente: {soggetto} {verbo}, reparto fermo.",
        ],
    },
    "Commerciale": {
        "soggetti": [
            "ordine cliente", "preventivo", "listino prezzi", "catalogo prodotti",
            "offerta commerciale", "spedizione", "reso merce",
            "consegna", "promozione", "contratto di fornitura",
            "campione gratuito", "scontistica", "trattativa",
            "lead commerciale", "piano vendite",
        ],
        "verbi": [
            "non è stato inviato", "contiene errori",
            "deve essere aggiornato", "è stato annullato",
            "non è stato ricevuto", "va modificato",
            "richiede approvazione", "è scaduto",
            "necessita revisione", "non corrisponde alla richiesta",
        ],
        "titoli": [
            "Problema con {soggetto}",
            "Richiesta aggiornamento {soggetto}",
            "Segnalazione su {soggetto}",
            "Modifica {soggetto}",
            "{soggetto} da rivedere",
            "Sollecito {soggetto}",
        ],
        "frasi_corpo": [
            "Il/la {soggetto} {verbo}. Si prega di intervenire.",
            "Gentilmente, il/la {soggetto} {verbo}. Attendo aggiornamento.",
            "Segnalo che il/la {soggetto} {verbo}.",
            "Il cliente ha segnalato che il/la {soggetto} {verbo}.",
            "Il/la {soggetto} {verbo}. Chiedo cortesemente una verifica.",
            "Buongiorno, il/la {soggetto} {verbo}. Urgente per il cliente.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Parole chiave per priorità
# ---------------------------------------------------------------------------

KEYWORDS_ALTA = [
    "bloccante", "urgente", "errore critico", "fermo", "impossibile lavorare",
    "guasto", "crash", "sistema down", "emergenza", "priorità massima",
]

KEYWORDS_MEDIA = [
    "sollecito", "scadenza", "ritardo", "attesa", "non conforme",
    "da rivedere", "anomalia", "tempestivo", "entro oggi",
]

FRASI_URGENZA_ALTA = [
    " Situazione bloccante, serve risposta immediata.",
    " È un errore critico che impedisce l'operatività.",
    " Urgente: il reparto è fermo.",
    " Impossibile lavorare senza risoluzione.",
    " Il sistema è down, emergenza.",
]

FRASI_URGENZA_MEDIA = [
    " Sollecito cortesemente una risposta.",
    " La scadenza è imminente, serve un riscontro.",
    " C'è un ritardo significativo da gestire.",
    " Attendo riscontro entro oggi se possibile.",
    " L'anomalia va risolta in tempi brevi.",
]


def genera_ticket(ticket_id: int, categoria: str, priorita_target: str) -> dict:
    """Genera un singolo ticket con la categoria e priorità specificate."""
    tmpl = TEMPLATES[categoria]
    soggetto = random.choice(tmpl["soggetti"])
    verbo = random.choice(tmpl["verbi"])

    titolo = random.choice(tmpl["titoli"]).format(soggetto=soggetto)
    corpo = random.choice(tmpl["frasi_corpo"]).format(soggetto=soggetto, verbo=verbo)

    # Aggiungi frasi di urgenza in base alla priorità target
    if priorita_target == "alta":
        corpo += random.choice(FRASI_URGENZA_ALTA)
    elif priorita_target == "media":
        corpo += random.choice(FRASI_URGENZA_MEDIA)

    return {
        "id": ticket_id,
        "title": titolo,
        "body": corpo,
        "category": categoria,
        "priority": priorita_target,
    }


def genera_dataset(n_ticket: int = 500) -> list[dict]:
    """Genera l'intero dataset bilanciato per categoria con distribuzione priorità."""
    categorie = list(TEMPLATES.keys())
    ticket_per_categoria = n_ticket // len(categorie)
    resto = n_ticket % len(categorie)

    dataset = []
    ticket_id = 1

    for i, cat in enumerate(categorie):
        n = ticket_per_categoria + (1 if i < resto else 0)

        # Distribuzione priorità: ~40% bassa, ~35% media, ~25% alta
        n_bassa = round(n * 0.40)
        n_media = round(n * 0.35)
        n_alta = n - n_bassa - n_media

        priorita_list = (
            ["bassa"] * n_bassa + ["media"] * n_media + ["alta"] * n_alta
        )
        random.shuffle(priorita_list)

        for priorita in priorita_list:
            dataset.append(genera_ticket(ticket_id, cat, priorita))
            ticket_id += 1

    random.shuffle(dataset)
    return dataset


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "tickets.csv")

    dataset = genera_dataset(500)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "body", "category", "priority"])
        writer.writeheader()
        writer.writerows(dataset)

    # Statistiche
    from collections import Counter
    cat_counts = Counter(t["category"] for t in dataset)
    pri_counts = Counter(t["priority"] for t in dataset)

    print(f"Dataset generato: {output_path}")
    print(f"Totale ticket: {len(dataset)}")
    print(f"Distribuzione categorie: {dict(cat_counts)}")
    print(f"Distribuzione priorità: {dict(pri_counts)}")


if __name__ == "__main__":
    main()
