from __future__ import annotations
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "data_afvoerlocaties_2024.csv"
OUT_CSV = ROOT / "outputs" / "data_afvoerlocaties_2024_met_manegeinfo.csv"


def _normalize_pc(s: str) -> str:
    """Zet postcode naar PC6: 1234AB (zonder spatie, hoofdletters)."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", "", str(s)).upper()


def _find_column(pattern: str, columns, error_msg: str) -> str:
    """Zoek kolom waarvan de naam het patroon bevat (case-insensitive)."""
    pattern = pattern.lower()
    matches = [c for c in columns if pattern in c.lower()]
    if not matches:
        raise ValueError(error_msg)
    return matches[0]


def load_main_dataset(path: Path) -> pd.DataFrame:
    """Lees jouw grote CSV en voeg een PC6-kolom toe."""
    df = pd.read_csv(path, sep=";")
    pc_col = _find_column(
        "postcode",
        df.columns,
        "Geen postcode-kolom gevonden in data_afvoerlocaties_2024.csv",
    )
    df["PC6"] = df[pc_col].astype(str).apply(_normalize_pc)

    # alleen échte NL PC6 behouden, rest krijgt None zodat join netjes is
    mask_pc6 = df["PC6"].str.match(r"^\d{4}[A-Z]{2}$")
    df.loc[~mask_pc6, "PC6"] = None
    return df


def load_all_manege_files(outputs_dir: Path) -> pd.DataFrame:
    """Lees alle *_manege.csv bestanden en maak één tabel met unieke PC6."""
    files = sorted(outputs_dir.glob("*_manege.csv"))
    if not files:
        raise RuntimeError("Geen *_manege.csv bestanden gevonden in outputs/")

    frames = []
    for f in files:
        print(f"[info] lees manege-bestand: {f.name}")
        df = pd.read_csv(f)  # deze zijn met komma gescheiden

        # Verwachte kolommen:
        # Postcode, Naam manege, Straat en huisnummer, Plaats,
        # Telefoonnummer (indien beschikbaar)  <-- hier zit nu jouw KVK in
        required = [
            "Postcode",
            "Naam manege",
            "Straat en huisnummer",
            "Plaats",
            "Telefoonnummer (indien beschikbaar)",
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Kolom {col!r} ontbreekt in {f}")

        df = df.copy()
        df["PC6"] = df["Postcode"].astype(str).apply(_normalize_pc)

        frames.append(
            df[
                [
                    "PC6",
                    "Naam manege",
                    "Straat en huisnummer",
                    "Plaats",
                    "Telefoonnummer (indien beschikbaar)",
                ]
            ]
        )

    all_manege = pd.concat(frames, ignore_index=True)

    # Als er meerdere maneges per postcode zijn:
    # neem gewoon de eerste (je kunt hier later ' | '.join(...) van maken als je wilt)
    all_manege = (
        all_manege.sort_values("Naam manege")
        .drop_duplicates(subset=["PC6"], keep="first")
        .reset_index(drop=True)
    )

    # Hernoem telefoon-kolom naar KVK, zoals jij wilt
    all_manege = all_manege.rename(
        columns={"Telefoonnummer (indien beschikbaar)": "KVK nummer"}
    )

    all_manege["KVK nummer"] = (
    all_manege["KVK nummer"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.strip()
    )

    return all_manege


def main():
    print("[info] hoofd-dataset inlezen...")
    df_main = load_main_dataset(DATA_CSV)
    print(f"[info] hoofd-dataset rijen: {len(df_main)}")

    print("[info] manege-bestanden inlezen...")
    df_manege = load_all_manege_files(ROOT / "outputs")
    print(f"[info] postcodes met manege-info: {len(df_manege)}")

    print("[info] join op PC6 (left join)...")
    df_merged = df_main.merge(df_manege, how="left", on="PC6")

    # Kolomnamen zo laten zoals jij ze in de manege-bestanden hebt:
    # 'Naam manege', 'Straat en huisnummer', 'Plaats', 'KVK nummer'

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUT_CSV, sep=";", index=False)
    print(f"[done] Geschreven: {OUT_CSV}")
    print(f"[done] Rijen totaal: {len(df_merged)}")


if __name__ == "__main__":
    main()
