import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup  # Voor fallback parsing als nodig
from pathlib import Path
import subprocess
import shutil
from playwright.sync_api import sync_playwright
import re

# Lees postcodes uit CSV (eerste kolom)
#df_postcodes = pd.read_csv('data_afvoerlocaties_2024.csv')
ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "data" / "data_afvoerlocaties_2024.csv"
OUTPUT_PATH = ROOT / "outputs" / "fnrs_manege_matches.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_postcodes = pd.read_csv(CSV, sep=';')

print("Kolommen in je CSV:")
for i, col in enumerate(df_postcodes.columns):
    print(f"  Kolom {i}: '{col}'")

# TWEEDE kolom = index 1
postcodes = (df_postcodes.iloc[:, 1]
             .dropna()
             .astype(str)
             .str.strip()
             .str.upper()          
             .tolist())

print(f"\nAantal postcodes gevonden: {len(postcodes)}")
print(f"Eerste 10 postcodes: {postcodes[:10]}")

results = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 1x naar zoekpagina
    page.goto("https://www.fnrs.nl/fnrs-bedrijven", wait_until="networkidle")

    for i, pc in enumerate(postcodes, start=1):
        print(f"{i}/{len(postcodes)} → {pc}", end=" ")

        try:
            # veld leegmaken + invullen
            field = page.get_by_placeholder("Jouw postcode")  # gebruik placeholder
            field.fill("")         # eerst leegmaken
            field.fill(pc)         # dan postcode invullen
            field.press("Enter")   # zoek starten
            page.wait_for_load_state("networkidle")

            # wacht op ofwel 'Geen bedrijven gevonden' of een resultaat
            page.wait_for_timeout(200)  # mini-pauze
            page.wait_for_load_state("networkidle")

            # check 'Geen bedrijven gevonden'
            if page.locator("text=Geen bedrijven gevonden").first.is_visible(timeout=2000):
                results.append([pc, "Geen FNRS-bedrijf", "", "", ""])
                print("→ geen")
                continue

            # eerste resultaat pakken
            first_result = page.locator("a[href*='/fnrs-bedrijven/']").first
            if first_result.count() == 0:
                results.append([pc, "Geen klikbaar resultaat", "", "", ""])
                print("→ geen klikbaar")
                continue

            # detailpagina openen
            first_result.click()
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("h1", timeout=5000)

            # --- detail uitlezen ---
            naam = page.locator("h1").inner_text().strip()

            adres_locator = page.locator("address, .fnrs-bedrijf__adres").first
            if adres_locator.count() > 0:
                adres_text = adres_locator.inner_text().strip()
            else:
                # fallback: hele pagina-tekst gebruiken
                adres_text = page.inner_text("body").strip()
            
            straat = ""
            plaats = ""
            gevonden_pc = None        
            
            if adres_text:
                # zoek éérste NL-postcode ergens in de tekst
                m = re.search(r"\b(\d{4})\s*([A-Z]{2})\b", adres_text)
                if m:
                    gevonden_pc = (m.group(1) + m.group(2)).upper()

                    # probeer de hele regel met de postcode te pakken
                    text = adres_text.replace("\r", "")
                    start_line = text.rfind("\n", 0, m.start())
                    end_line = text.find("\n", m.end())
                    if end_line == -1:
                        end_line = len(text)
                    line = text[start_line + 1:end_line].strip()  # regel met '1234 AB PLAATS'

                    # vorige regel nemen als straat+huisnummer (vaak zo opgebouwd)
                    prev_start = text.rfind("\n", 0, start_line)
                    if start_line != -1:
                        prev_line = text[prev_start + 1:start_line].strip()
                        if prev_line:
                            straat = prev_line

                    # plaats: stuk ná postcode in dezelfde regel
                    after = line[m.end() - (start_line + 1):].strip()
                    if after:
                        # neem laatste "woord" als plaatsnaam (ruw maar werkt vaak)
                        plaats = after.split()[-1] 

            # --- postcode match check ---
            gezochte_pc = pc.replace(" ", "").upper()
            if not gevonden_pc or gevonden_pc != gezochte_pc:
                # geen postcode in adres, of postcode ≠ gezochte postcode → geen match
                results.append([pc, "Geen FNRS-bedrijf (postcode mismatch)", "", "", ""])
                print("→ postcode mismatch of geen postcode in adres")
                # terug naar resultaten/zoekpagina voor volgende postcode
                page.go_back(wait_until="networkidle")
                continue

            tel = page.locator("a[href^='tel:']").first
            telefoon = tel.inner_text().strip() if tel.count() > 0 else ""

            results.append([pc, naam, straat, plaats, telefoon])
            print(f"→ {naam} ({plaats})")

            # terug naar resultatenpagina i.p.v. opnieuw goto
            page.go_back(wait_until="networkidle")

        except Exception as e:
            print(f"→ fout ({e})")
            results.append([pc, "Fout", "", "", ""])
            # probeer schoon op zoekpagina te komen
            page.goto("https://www.fnrs.nl/fnrs-bedrijven", wait_until="networkidle")


# ----------------------------- OPSLAAN (zelfde formaat als je andere scraper) -----------------------------
pd.DataFrame(results, columns=[
    "Postcode",
    "Naam manege",
    "Straat en huisnummer",
    "Plaats",
    "Telefoonnummer (indien beschikbaar)"
]).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\nKlaar! Resultaten opgeslagen in:\n{OUTPUT_PATH}")