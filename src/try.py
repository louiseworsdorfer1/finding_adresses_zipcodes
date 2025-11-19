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
PHONE_RE = re.compile(r"(?:\+31|0031|0)[1-9]\d(?:[\s-]?\d){7,8}\b")  # NL-telefoons

PROVINCE_SLUGS = {
    "groningen": ["groningen"],
    "friesland": ["friesland", "fryslân", "fryslan"],
    "drenthe": ["drenthe"],
    "overijssel": ["overijssel", "overijsel"],
    "flevoland": ["flevoland"],
    "gelderland": ["gelderland"],
    "utrecht": ["utrecht"],
    "noord-holland": ["noord-holland", "noord holland"],
    "zuid-holland": ["zuid-holland", "zuid holland"],
    "zeeland": ["zeeland"],
    "noord-brabant": ["noord-brabant", "noord brabant"],
    "limburg": ["limburg"],
}

def _normalize_prov_list(prov_arg: Optional[str]) -> Optional[Set[str]]:
    """Zet 'Zuid-Holland, Utrecht' om naar set canonical keys: {'zuid-holland','utrecht'}"""
    if not prov_arg:
        return None
    out = set()
    for raw in prov_arg.split(","):
        k = raw.strip().lower()
        # canonicaliseer naar de dict-key als het een variant is
        for key, variants in PROVINCE_SLUGS.items():
            if k == key or k in variants:
                out.add(key)
                break
        else:
            # geen match → gebruik zoals ingevoerd (valt terug op tekst-zoek)
            out.add(k)
    return out

def _page_mentions_province(soup: BeautifulSoup, url: str, allowed: Set[str]) -> bool:
    """Check of een 'plaats'-pagina bij één van de gevraagde provincies hoort.
    We zoeken op:
      1) URL-tokens (slugs)
      2) Breadcrumbs/teksten op de pagina
    """
    url_l = url.lower()
    for key in allowed:
        tokens = PROVINCE_SLUGS.get(key, [key])
        if any(tok in url_l for tok in tokens):
            return True

    # Tekstzoek (fallback)
    txt = soup.get_text(" ", strip=True).lower()
    for key in allowed:
        tokens = PROVINCE_SLUGS.get(key, [key])
        if any(tok in txt for tok in tokens):
            return True
    return False


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

def iter_place_links(root: BeautifulSoup, provinces: Optional[Set[str]] = None) -> Iterable[str]:
    # Geeft plaats-URLs terug, optioneel gefilterd op provincie door
    # het dichtsbijzijnde voorgaande kopje te gebruiken als provincie-label.
    for a in root.find_all("a", href=True):
        href = a["href"]
        href = urljoin(BASE_URL, href) if href.startswith("/") else href
        if "bedrijvenregister.nl" not in urlparse(href).netloc or href == BASE_URL:
            continue

        # zoek dichtstbijzijnde kop boven deze link
        hdr = a.find_previous(["h2", "h3", "h4", "strong", "b"])
        prov = _canonicalize_province_label(hdr.get_text(strip=True).lower() if hdr else "")

        if provinces:
            # alleen links onder de gevraagde provincie-kopjes
            if prov not in provinces:
                continue
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

    seen, uniq = set(), []
    for b in blocks:
        txt = b.get_text(strip=True)
        if not txt:
            continue
        key = b.get("id") or txt[:160]
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq

def parse_block_for_fields(block: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
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

def _canonicalize_province_label(text: str) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.strip().lower())
    for key, variants in PROVINCE_SLUGS.items():
        if t == key or any(t == v for v in variants):
            return key
        # ook tolerant voor kopjes met extra tekst zoals "Zuid-Holland"
        if any(v in t for v in [key, *variants]):
            return key
    return None

def iter_company_links_on_place(soup: BeautifulSoup, place_url: str) -> list[str]:
    """Return absolute URLs to company detail pages for this place."""
    place_slug = urlparse(place_url).path.strip("/").split("/")[-1].lower()
    out = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin(BASE_URL, href)
        p = urlparse(href).path.lower()

        # Detail pages for this place look like "/<plaats>/<bedrijfsnaam>"
        # So require "/<place_slug>/" somewhere in the path
        if f"/{place_slug}/" in p:
            out.append(href)

    # dedupe, keep order
    return list(dict.fromkeys(out))


def parse_company_detail(soup: BeautifulSoup, url: Optional[str] = None):
    def clean(txt: str) -> str:
        return " ".join(txt.split()).strip()

    page_text = soup.get_text(" ", strip=True)

    # --- Naam ---
    name = None
    h = soup.select_one("h1") or soup.select_one("h2")
    if h and h.get_text(strip=True):
        name = h.get_text(strip=True)
    if not name:
        h = soup.find(["strong", "b"])
        if h and h.get_text(strip=True):
            name = h.get_text(strip=True)

    if (not name) or (name.strip().lower() == "telefonisch contact"):
        if url:  # voorkom crash als url None is
            slug = urlparse(url).path.strip("/").split("/")[-1]
            name_from_url = re.sub(r"[-_]+", " ", slug).strip()
            if name_from_url:
                name = name_from_url.title()

    # --- Adres via 'Hoofdvestiging' blok ---
    street = plaats = postcode = None

    hoofd_node = None
    # Vind het element dat 'Hoofdvestiging' bevat
    for el in soup.find_all(text=re.compile(r"^\s*Hoofdvestiging\s", re.I)):
        hoofd_node = el.parent
        break
    if not hoofd_node:
        # sommige pagina's hebben 'Hoofdvestiging <Bedrijfsnaam>'
        for el in soup.find_all(text=re.compile(r"^\s*Hoofdvestiging\b", re.I)):
            hoofd_node = el.parent
            break

    if hoofd_node:
        # Neem de eerstvolgende niet-lege 4–6 tekstregels na de titel
        lines: list[str] = []
        cur = hoofd_node
        # loop door de volgende nodes en verzamel korte tekstregels
        for _ in range(12):
            cur = cur.find_next(string=True)
            if not cur:
                break
            t = clean(str(cur))
            if not t:
                continue
            # Stop als we een nieuwe sectie/header raken
            if re.search(r"^(Geregistreerde handelsnamen|Opgegeven bedrijfsactiviteit|KVK inschrijving|Uittreksel|Eigenareninformatie|Download|Telefoon)", t, re.I):
                break
            lines.append(t)
            if len(lines) >= 6:
                break

        # Zoek in deze regels naar een regel met postcode + plaats
        zip_line = next((ln for ln in lines if re.search(r"\b\d{4}\s?[A-Z]{2}\b", ln)), None)
        if zip_line:
            m = re.search(r"\b(\d{4})\s?([A-Z]{2})\b\s+(.+)", zip_line)
            if m:
                postcode = m.group(1) + m.group(2)
                plaats = clean(m.group(3))

            # Straatregel is meestal de regel direct vóór de zip-regel
            idx = lines.index(zip_line)
            if idx > 0:
                street = clean(lines[idx - 1])

    # --- Fallback adres uit intro-zin 'gevestigd op het adres: ... te Plaats' ---
    if not postcode or not street or not plaats:
        intro = None
        # pak eerste langere paragraaf in bovenstuk
        for p in soup.select("p"):
            t = clean(p.get_text(" ", strip=True))
            if t and len(t) > 30:
                intro = t
                break
        if not intro:
            intro = page_text

        # postcode uit de tekst
        m = PC_RE.search(intro)
        if m:
            postcode = postcode or (m.group(1) + m.group(2))
        # straat – heuristiek: stukje voor postcode (laatste token met cijfers)
        if m:
            before = clean(intro[:m.start()])
            # probeer laatste 'woordblok' met cijfer = straat + nr
            m_st = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ' .-]+\d+[A-Za-z0-9\- ]*)$", before)
            if m_st:
                street = street or clean(m_st.group(1))
        # plaats – na postcode kijk naar ' te PLAATS' of direct na postcode
        if m:
            after = clean(intro[m.end():])
            m_pl = re.search(r"\bte\s+([A-Za-zÀ-ÖØ-öø-ÿ' .-]+)", after, re.I)
            if m_pl:
                plaats = plaats or clean(m_pl.group(1).split()[0])

    # --- Telefoon (alleen in sectie met 'Telefoonnummer') ---
    phone = None
    tel_container = None
    for el in soup.find_all(text=re.compile(r"telefoonnummer", re.I)):
        tel_container = el.parent
        break
    if tel_container:
        t = tel_container.get_text(" ", strip=True)
        m = PHONE_RE.search(t)
        if m:
            phone = m.group(0)

    if not phone:
        m_kvk = re.search(r"kvk\s*nummer[:\s]*([0-9]{6,10})", page_text, re.I)
        if m_kvk:
            phone = m_kvk.group(1)

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
def scrape_and_match(allowed_postcodes: Set[str], max_places: int = 0, provinces: Optional[Set[str]] = None, no_dataset_filter: bool = False) -> List[Row]:
    """
    Crawlt de categoriepagina, bezoekt subpagina's (plaatsen) en verzamelt bedrijfsblokken.
    Filtert vervolgens op:
      - postcode in allowed_postcodes (NL PC6)
      - (optioneel) naam bevat manege/rijschool/pensionstal/...
    """
    root = fetch(BASE_URL, REQUEST_DELAY)
    if not root:
        raise RuntimeError("Kon hoofdpagina niet laden.")

    rows: List[Row] = []
    seen_places = 0

    place_links = list(iter_place_links(root, provinces=provinces))
    print(f"[dbg] plaats-links na provinciefilter: {len(place_links)}")

    for idx, place_url in enumerate(place_links, 1):
        print(f"[dbg] visiting {idx}/{len(place_links)}: {place_url}", flush=True)

        soup = fetch(place_url, REQUEST_DELAY)
        if not soup:
            print("[dbg] fetch failed, skip", flush=True)
            continue

        comp_links = iter_company_links_on_place(soup, place_url)
        print(f"[dbg] found {len(comp_links)} company links on place page", flush=True)
        print("[dbg] sample company links:", comp_links[:3], flush=True)

        hits = 0
        for comp_url in comp_links:
            csoup = fetch(comp_url, REQUEST_DELAY)
            if not csoup:
                continue

            name, street, pc, plaats, phone = parse_company_detail(csoup, comp_url)
            if not pc:
                continue
            hits += 1
            tag = "IN-DATASET" if (pc in allowed_postcodes) else "NOT-IN-DATASET"
            #print(f"[dbg_pc] {pc}  {tag}  name={name!r}", flush=True)

            if (not no_dataset_filter) and (pc not in allowed_postcodes):
                continue

            rows.append(Row(
                Postcode=pc,
                Naam=name or "",
                Straat_en_huisnummer=street or "",
                Plaats=plaats or "",
                Telefoonnummer=phone or "",
            ))
        print(f"[dbg] postcodes gevonden op pagina: {hits}", flush=True)
 
        seen_places += 1
        if max_places and seen_places > max_places:
            break 

    # Deduplicate op (Postcode, Naam)
    dedup = {}
    for r in rows:
        key = (r.Postcode, r.Naam.strip().lower())
        if key not in dedup:
            dedup[key] = r
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

    ap = argparse.ArgumentParser(description="Zoek bedrijven (SBI 0143) die matchen met jouw NL PC6 postcodes.")
    ap.add_argument("--max_places", type=int, default=0,
                    help="Maximum # plaats-pagina's om te crawlen (0 = alles). Handig voor testen.")
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY,
                    help="Vertraging tussen requests (seconden).")
    ap.add_argument("--output", type=str, default=str(ROOT / "outputs" / "manege_matches_bedrijvenregister.csv"),
                    help="Pad naar output CSV.")
    ap.add_argument("--province", type=str, default="", help="Filter op provincie (bijv. 'Zuid-Holland' of meerdere: 'Utrecht, Zuid-Holland').")
    ap.add_argument("--no_dataset_filter", action="store_true",
                help="Negeer allowed_postcodes en bewaar alle gevonden postcodes.")
    args = ap.parse_args()

    # Update delay desgewenst
    REQUEST_DELAY = float(args.delay)

    # 1) Postcodes inlezen zoals jouw project dat doet
    allowed = load_allowed_postcodes_from_project_csv(CSV)
    print(f"[info] #unieke NL PC6 postcodes in jouw data: {len(allowed)}")

    prov_set = _normalize_prov_list(args.province)
    if prov_set:
        print(f"[info] Provinciefilter actief: {', '.join(sorted(prov_set))}")

    # 3) Scrapen + matchen
    rows = scrape_and_match(allowed_postcodes=allowed,
                            max_places=args.max_places,
                            provinces=prov_set,
                            no_dataset_filter=args.no_dataset_filter)

    # 4) Opslaan in de gevraagde kolommen
    out_csv = Path(args.output)
    save_rows_csv(rows, out_csv)
    print(f"[done] Geschreven: {out_csv} (records: {len(rows)})")
