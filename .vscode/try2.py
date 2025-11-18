# src/scrape_manege_bedrijvenregister.py
from __future__ import annotations

"""
Scraper voor https://www.bedrijvenregister.nl/fokken-en-houden-van-paarden-en-ezels
- Leest jouw CSV (data/data_afvoerlocaties_2024.csv) op dezelfde manier als in perceel_radius.py
- Normaliseert postcodes naar PC6 (1234AB) en filtert op NL
- Crawler bezoekt alle 'plaats'-pagina's onder de categorie en probeert bedrijfskaarten te parsen
- Schrijft een CSV met: Postcode | Naam manege | Straat en huisnummer | Plaats | Telefoonnummer (indien beschikbaar) | Website (indien beschikbaar)
"""

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "data" / "data_afvoerlocaties_2024.csv"

# === Web settings ===
BASE_URL = "https://www.bedrijvenregister.nl/fokken-en-houden-van-paarden-en-ezels"
UA = "Mozilla/5.0 (compatible; manege-scraper/1.0; +https://example.local)"
REQUEST_DELAY = 1.0  

# === Regexen ===
PC_RE = re.compile(r"\b(\d{4})\s?([A-Z]{2})\b")                 # 1234AB / 1234 AB
PHONE_RE = re.compile(r"(?:(?:\+|00)31|0)\s?\d(?:[\s-]?\d){8,10}")  # NL-telefoons

# ========== Postcode-inlezen ==============
def _normalize_pc(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", "", s.upper())

def _detect_country(pc_clean: str) -> str | None:
    # NL PC6: 1234AB | BE: 4 digits | DE: 5 digits
    if re.fullmatch(r"\d{4}[A-Z]{2}", pc_clean):
        return "NL6"
    if re.fullmatch(r"\d{4}", pc_clean):
        return "BE"
    if re.fullmatch(r"\d{5}", pc_clean):
        return "DE"
    return None

def _find_column(pattern: str, columns, error_msg: str) -> str:
    matches = [c for c in columns if pattern in c.lower()]
    if not matches:
        raise ValueError(error_msg)
    return matches[0]

def load_allowed_postcodes_from_project_csv(csv_path: Path) -> Set[str]:
    """Leest jouw CSV en geeft een set terug met NL PC6 postcodes (1234AB)."""
    df = pd.read_csv(csv_path, sep=";")
    postcode_col = _find_column("postcode", df.columns, "Geen postcode-kolom gevonden (verwacht iets met 'postcode').")
    df["PC_CLEAN"] = df[postcode_col].astype(str).apply(_normalize_pc)
    df["PC_TYPE"] = df["PC_CLEAN"].apply(_detect_country)
    # Neem alleen NL PC6
    df = df[df["PC_TYPE"].eq("NL6")].copy()
    # Uniek PC6
    pcs = set(df["PC_CLEAN"].dropna().unique().tolist())
    return pcs

# ========== Web-scraper helpers ==========
def fetch(url: str, delay: float = REQUEST_DELAY) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if resp.status_code != 200:
            return None
        time.sleep(delay)
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException:
        return None

def iter_place_links(root: BeautifulSoup) -> Iterable[str]:
    """
    Zoekt alle links naar subpagina's (plaatsen/provincies) onder deze categorie.
    We filteren op domein en sluiten de hoofdpagina zelf uit.
    """
    for a in root.find_all("a", href=True):
        href = a["href"]
        href = urljoin(BASE_URL, href) if href.startswith("/") else href
        if "bedrijvenregister.nl" in urlparse(href).netloc and href != BASE_URL:
            # lichte heuristiek: veel plaats-pagina's zijn directe subpaden
            yield href

def extract_company_blocks(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """
    Bedrijven kunnen in verschillende container-elementen staan.
    We proberen meerdere selectors en dedupliceren op tekst.
    """
    blocks = []
    blocks.extend(soup.select("div.bedrijf"))
    blocks.extend(soup.select("article"))
    blocks.extend(soup.select("li"))
    blocks.extend(soup.select("div"))

    seen = set()
    uniq = []
    for b in blocks:
        txt = b.get_text(strip=True)
        if not txt:
            continue
        key = b.get("id") or txt[:160]
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq

def parse_block_for_fields(block: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Probeer uit een 'bedrijfsblok' te halen:
      - name
      - street&nr
      - postcode (PC6)
      - plaats
      - phone
    We gebruiken regex en heuristieken rondom de gevonden postcode.
    """
    txt = " ".join(block.get_text(" ", strip=True).split())
    if not txt:
        return None, None, None, None, None

    # 1) Naam: eerst linktekst, dan headings
    name = None
    a = block.find("a")
    if a and a.get_text(strip=True):
        name = a.get_text(strip=True)
    if not name:
        for tag in ["h1", "h2", "h3", "strong", "b"]:
            h = block.find(tag)
            if h and h.get_text(strip=True):
                name = h.get_text(strip=True)
                break

    # 2) Postcode/plaats/straat
    street = plaats = postcode = None
    m = PC_RE.search(txt)
    if m:
        postcode = m.group(1) + m.group(2)    # 1234AB
        before = txt[:m.start()].strip(" ,;-")
        after  = txt[m.end():].strip(" ,;-")
        # Heuristiek: vóór postcode staat vaak straat+nr
        street = before.split("  ")[-1].strip() if before else None
        # Heuristiek: eerste woord na postcode is vaak plaats
        plaats = after.split(" ")[0].strip(",") if after else None

    # 3) Telefoon
    phone = None
    m2 = PHONE_RE.search(txt)
    if m2:
        phone = m2.group(0).strip()

    return name, street, postcode, plaats, phone

# ========== Output-structuur ==========
@dataclass
class Row:
    Postcode: str
    Naam: str
    Straat_en_huisnummer: str
    Plaats: str
    Telefoonnummer: str

# ========== Hoofdlogica ==========
def scrape_and_match(
    allowed_postcodes: Set[str],
    offset: int = 0,
    limit: int = 0,
) -> List[Row]:
    """
    Crawlt de categoriepagina, bezoekt subpagina's (plaatsen) en verzamelt bedrijfsblokken.
    Met offset/limit kun je het netjes in stukken hakken.
    """
    root = fetch(BASE_URL, REQUEST_DELAY)
    if not root:
        raise RuntimeError("Kon hoofdpagina niet laden.")

    rows: List[Row] = []
    processed = 0          # hoeveel plaatsen we écht hebben verwerkt
    skipped   = 0          # hoeveel we hebben overgeslagen door offset

    # Eerst alle plaats-URL's verzamelen
    all_place_urls = list(dict.fromkeys(iter_place_links(root)))  # dedup + behoud volgorde
    print(f"[info] Totaal gevonden plaats-pagina's: {len(all_place_urls)}")

    for place_url in all_place_urls:
        # --- offset overslaan ---
        if skipped < offset:
            skipped += 1
            continue

        # --- limit bereikt? ---
        processed += 1
        if limit and processed > limit:
            print(f"[info] Limiet van {limit} bereikt → stoppen")
            break

        print(f"[{processed + offset}/{len(all_place_urls)}] Crawlen: {place_url}")
        soup = fetch(place_url, REQUEST_DELAY)
        if not soup:
            print(f"[warn] Kon pagina niet laden: {place_url}")
            continue

        for block in extract_company_blocks(soup):
            name, street, postcode, plaats, phone = parse_block_for_fields(block)
            if not postcode or postcode not in allowed_postcodes:
                continue

            rows.append(
                Row(
                    Postcode=postcode,
                    Naam=name or "",
                    Straat_en_huisnummer=street or "",
                    Plaats=plaats or "",
                    Telefoonnummer=phone or "",
                )
            )

    # Dedupliceren (zelfde als voorheen)
    dedup = {}
    for r in rows:
        key = (r.Postcode, r.Naam.strip().lower())
        if key not in dedup:
            dedup[key] = r
    print(f"[info] Klaar met dit chunk → {len(dedup)} unieke bedrijven gevonden")
    return list(dedup.values())


def save_rows_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Postcode",
            "Naam manege",
            "Straat en huisnummer",
            "Plaats",
            "Telefoonnummer (indien beschikbaar)",
        ])
        for r in rows:
            w.writerow([
                r.Postcode or "n.v.t.",
                r.Naam or "n.v.t.",
                r.Straat_en_huisnummer or "n.v.t.",
                r.Plaats or "n.v.t.",
                r.Telefoonnummer or "n.v.t.",
            ])

# ========== CLI ==========
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Zoek maneges die matchen met jouw NL PC6 postcodes – nu met chunking!"
    )
    ap.add_argument(
        "--offset", type=int, default=0,
        help="Skip de eerste X plaats-pagina's (voor chunking, bijv. 50)"
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Maximum aantal plaats-pagina's om te crawlen (0 = geen limiet)"
    )
    ap.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help="Vertraging tussen requests (seconden)."
    )
    ap.add_argument(
        "--output", type=str,
        default=str(ROOT / "outputs" / "manege_matches_bedrijvenregister.csv"),
        help="Pad naar output CSV."
    )
    args = ap.parse_args()

    # Update delay desgewenst
    REQUEST_DELAY = float(args.delay)

    # 1) Postcodes inlezen
    allowed = load_allowed_postcodes_from_project_csv(CSV)
    print(f"[info] #unieke NL PC6 postcodes in jouw data: {len(allowed)}")

    # 2) Scrapen + matchen met chunking
    rows = scrape_and_match(
        allowed_postcodes=allowed,
        offset=args.offset,
        limit=args.limit,
    )

    # 3) Opslaan
    out_csv = Path(args.output)
    save_rows_csv(rows, out_csv)
    print(f"[done] Geschreven: {out_csv} (records: {len(rows)})")