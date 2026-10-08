"""
DHA Lahore Phase 8 listings -> clean CSV (+ RAG-ready text per listing)

Usage:
    pip install requests pandas
    python clean_dha_listings.py                 # fetches live API
    python clean_dha_listings.py --input raw.json  # uses a saved JSON file

Outputs:
    dha_phase8_clean.csv   -> one clean row per listing
    dha_phase8_docs.jsonl  -> {"id", "text", "metadata"} per listing, ready for Qdrant/LangChain
"""

import argparse
import json
import re

import pandas as pd
import requests

API_URL = "https://api.dhaconnects.com/api/plots/public/dha-lahore-phase-8/listings/"
IMAGE_BASE = "https://api.dhaconnects.com/"
MARLA_PER_KANAL = 20


# ---------- fetch ----------
def load_raw(input_path=None):
    if input_path:
        with open(input_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        resp = requests.get(API_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError("API returned success=false")
    return payload["data"], payload.get("meta", {})


# ---------- helpers ----------
def to_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def area_in_marla(area, unit):
    a = to_float(area)
    if a is None:
        return None
    unit = (unit or "").strip().lower()
    return a * MARLA_PER_KANAL if unit == "kanal" else a


def marla_from_title(title):
    """Pull size stated in the title, e.g. '26 Marla', '1 Kanal', '1+1 ... Kanal'."""
    t = title.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*marla", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*kanal", t)
    if m:
        return float(m.group(1)) * MARLA_PER_KANAL
    return None


def clean_block(sector, block):
    raw = sector or block or ""
    raw = re.sub(r"(?i)^block\s*", "", raw.strip())
    raw = raw.upper().replace(" ", "")
    raw = re.sub(r"^CCA(\d)$", r"CCA-\1", raw)   # CCA3 -> CCA-3
    return raw


def plot_numbers(plot_clean):
    """'104-105' -> ['104','105'], '342 & 343' -> ['342','343'], '1329+30' -> ['1329','1330']."""
    parts = [p.strip() for p in re.split(r"\s*(?:-|&|\+|,|\band\b)\s*", str(plot_clean)) if p.strip()]
    if len(parts) < 2 or not all(p.isdigit() for p in parts):
        return [str(plot_clean)]
    first = parts[0]
    out = [first]
    for p in parts[1:]:
        # shorthand like 1329+30 means 1330
        out.append(first[: len(first) - len(p)] + p if len(p) < len(first) else p)
    return out


def fix_multi_plot_area(df):
    """
    For pair/multi-plot ads, the API sometimes stores ONE plot's size instead of the total.
    Multiply by the plot count only when we have evidence the area is per-plot:
      a) the title states a size and the area equals it (e.g. '1 Kanal Pair' with area 1 kanal), or
      b) another listing in the same block for one of these plot numbers has the same area.
    Otherwise leave it alone.
    """
    single = {
        (r.block_clean, r.plot_clean): r.area_marla
        for r in df.itertuples()
        if len(plot_numbers(r.plot_clean)) == 1
    }
    adjusted = []
    for i, r in df.iterrows():
        nums = plot_numbers(r.plot_clean)
        n = len(nums)
        is_per_plot = False
        if n > 1 and pd.notna(r.area_marla):
            if pd.notna(r.title_marla) and abs(r.title_marla - r.area_marla) <= 1:
                is_per_plot = True
            elif any(abs(single.get((r.block_clean, p), -99) - r.area_marla) <= 1 for p in nums):
                is_per_plot = True
        if is_per_plot:
            df.at[i, "area_marla"] = r.area_marla * n
        adjusted.append(is_per_plot)
    df["plot_count"] = df["plot_clean"].apply(lambda p: len(plot_numbers(p)))
    df["flag_area_adjusted"] = adjusted
    return df


def stated_phase(title):
    m = re.search(r"(?i)phase\s*(\d+)", title)
    return int(m.group(1)) if m else None


def price_label(pkr):
    if pkr is None:
        return "N/A"
    if pkr >= 1e7:
        return f"{pkr / 1e7:.2f} crore".replace(".00", "")
    return f"{pkr / 1e5:.1f} lakh".replace(".0", "")


# ---------- clean ----------
def clean(listings):
    df = pd.DataFrame(listings)

    df["block_clean"] = df.apply(lambda r: clean_block(r.get("sector"), r.get("block")), axis=1)
    df["plot_clean"] = df["plot"].astype(str).str.replace(r"(?i)plot\s*(no\.)?\s*", "", regex=True).str.strip()
    # "V 1165" in block V -> "1165"
    df["plot_clean"] = df.apply(
        lambda r: re.sub(rf"(?i)^{re.escape(r.block_clean)}[\s-]+", "", r.plot_clean), axis=1)
    df["price_pkr"] = df["price"].apply(to_float)
    df["area_marla"] = df.apply(lambda r: area_in_marla(r["area"], r["areaUnit"]), axis=1)
    df["title_marla"] = df["title"].apply(marla_from_title)
    df = fix_multi_plot_area(df)
    df["price_per_marla"] = (df["price_pkr"] / df["area_marla"]).round(0)
    df["price_label"] = df["price_pkr"].apply(price_label)
    df["thumbnail_url"] = df["thumbnailImage"].apply(lambda p: IMAGE_BASE + p if isinstance(p, str) and p else None)
    df["listing_url_slug"] = df["slug"]

    # land use: trust subType over landUse when they disagree
    df["land_use_clean"] = df["subType"].map(
        {"Residential Plot": "Residential", "Commercial Plot": "Commercial"}
    ).fillna(df["landUse"])

    # plots don't have rooms
    is_plot = df["type"].eq("Plots")
    df["bedrooms_clean"] = df["bedrooms"].where(~is_plot, None)
    df["bathrooms_clean"] = df["bathrooms"].where(~is_plot, None)

    # ---- quality flags (kept, not deleted, so you can decide) ----
    # title size is per plot for pair ads, so compare against the per-plot area
    per_plot_area = df["area_marla"] / df["plot_count"]
    df["flag_area_mismatch"] = (
        df["title_marla"].notna()
        & ((df["title_marla"] - df["area_marla"]).abs() > 1)
        & ((df["title_marla"] - per_plot_area).abs() > 1)
    )
    df["flag_landuse_mismatch"] = df["land_use_clean"] != df["landUse"]
    df["stated_phase"] = df["title"].apply(stated_phase)
    df["flag_other_phase"] = df["stated_phase"].notna() & (df["stated_phase"] != 8)

    # ---- duplicates: same block + plot -> keep the cheapest, record how many ----
    df = df.sort_values("price_pkr")
    df["duplicate_count"] = df.groupby(["block_clean", "plot_clean"])["slug"].transform("count")
    df["price_min_pkr"] = df.groupby(["block_clean", "plot_clean"])["price_pkr"].transform("min")
    df["price_max_pkr"] = df.groupby(["block_clean", "plot_clean"])["price_pkr"].transform("max")
    df = df.drop_duplicates(subset=["block_clean", "plot_clean"], keep="first")

    cols = [
        "title", "block_clean", "plot_clean", "type", "subType", "land_use_clean",
        "area_marla", "area", "areaUnit", "price_pkr", "price_label", "price_per_marla",
        "price_min_pkr", "price_max_pkr", "duplicate_count",
        "bedrooms_clean", "bathrooms_clean", "adType", "lat", "lng",
        "thumbnail_url", "listing_url_slug",
        "plot_count", "stated_phase",
        "flag_area_mismatch", "flag_area_adjusted", "flag_landuse_mismatch", "flag_other_phase",
    ]
    return df[cols].sort_values(["block_clean", "price_pkr"]).reset_index(drop=True)


# ---------- RAG docs ----------
def to_docs(df):
    docs = []
    for i, r in df.iterrows():
        size = f"{r.area_marla:g} marla" if pd.notna(r.area_marla) else "unknown size"
        if pd.notna(r.area_marla) and r.area_marla % MARLA_PER_KANAL == 0:
            size += f" ({r.area_marla / MARLA_PER_KANAL:g} kanal)"
        if r.plot_count > 1:
            size += f" total for {r.plot_count} adjoining plots ({r.area_marla / r.plot_count:g} marla each)"
        first = f"{r.subType} for sale in DHA Lahore Phase 8, Block {r.block_clean}, Plot {r.plot_clean}."
        if r.flag_other_phase:
            first = (f"{r.subType} for sale, Block {r.block_clean}, Plot {r.plot_clean}. Listed under "
                     f"DHA Lahore Phase 8, but the ad title says Phase {int(r.stated_phase)}, so the phase is uncertain.")
        parts = [
            first,
            f"Size: {size}. Land use: {r.land_use_clean}.",
            f"Price: PKR {r.price_pkr:,.0f} ({r.price_label}), about PKR {r.price_per_marla:,.0f} per marla.",
        ]
        if r.duplicate_count > 1:
            parts.append(
                f"Listed {r.duplicate_count} times with prices from {price_label(r.price_min_pkr)} "
                f"to {price_label(r.price_max_pkr)}."
            )
        if pd.notna(r.bedrooms_clean):
            parts.append(f"Bedrooms: {r.bedrooms_clean}, bathrooms: {r.bathrooms_clean}.")
        if r.flag_area_mismatch:
            parts.append("Note: the ad title states a different size than the listed area; confirm with the seller.")
        if r.flag_area_adjusted:
            parts.append("Note: the listing gave one plot's size; the total was calculated for all plots.")
        if r.flag_other_phase:
            parts.append("Confirm the phase with the seller before relying on this listing.")
        parts.append(f"Ad title: {r.title}")

        meta = {
            k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in r.items()
        }
        docs.append({"id": i, "text": " ".join(parts), "metadata": meta})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="saved raw JSON instead of calling the API")
    ap.add_argument("--csv", default="dha_phase8_clean.csv")
    ap.add_argument("--docs", default="dha_phase8_docs.jsonl")
    args = ap.parse_args()

    listings, meta = load_raw(args.input)
    df = clean(listings)
    df.to_csv(args.csv, index=False)

    with open(args.docs, "w", encoding="utf-8") as f:
        for d in to_docs(df):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"API meta: total={meta.get('total')} matched={meta.get('matched')} unmatched={meta.get('unmatched')}")
    print(f"Raw listings: {len(listings)}  ->  unique after dedupe: {len(df)}")
    print(f"Flags: area mismatch={df.flag_area_mismatch.sum()}, pair area fixed={df.flag_area_adjusted.sum()}, "
          f"land-use fixed={df.flag_landuse_mismatch.sum()}, other phase={df.flag_other_phase.sum()}")
    print(f"Saved {args.csv} and {args.docs}")


if __name__ == "__main__":
    main()
