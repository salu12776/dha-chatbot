import time
import clean_dha_listings as cdl

_CACHE = {"df": None, "at": 0}
CACHE_SECONDS = 600   # 10 min tak dobara fetch nahi

def get_listings(input_path=None):
    if _CACHE["df"] is None or time.time() - _CACHE["at"] > CACHE_SECONDS:
        listings, _ = cdl.load_raw(input_path)
        _CACHE["df"] = cdl.clean(listings)
        _CACHE["at"] = time.time()
    return _CACHE["df"]

def search_listings_impl(block=None, plot=None, land_use=None, property_type=None,
                         min_price_crore=None, max_price_crore=None,
                         min_marla=None, max_marla=None,
                         sort_by="price_asc", limit=10, input_path=None):
    df = get_listings(input_path).copy()
    if block:
        df = df[df.block_clean == cdl.clean_block(block, None)]
    if plot:
        df = df[df.plot_clean.astype(str).str.contains(str(plot), regex=False)]
    if land_use:
        df = df[df.land_use_clean.str.lower() == land_use.lower()]
    if property_type:
        df = df[df.type.str.lower().str.startswith(property_type.lower()[:4])]
    if min_price_crore is not None:
        df = df[df.price_pkr >= min_price_crore * 1e7]
    if max_price_crore is not None:
        df = df[df.price_pkr <= max_price_crore * 1e7]
    if min_marla is not None:
        df = df[df.area_marla >= min_marla - 0.5]
    if max_marla is not None:
        df = df[df.area_marla <= max_marla + 0.5]
    order = {"price_asc": ("price_pkr", True), "price_desc": ("price_pkr", False),
             "size_asc": ("area_marla", True), "size_desc": ("area_marla", False),
             "price_per_marla_asc": ("price_per_marla", True)}
    col, asc = order.get(sort_by, ("price_pkr", True))
    df = df.sort_values(col, ascending=asc)
    total = len(df)
    if total == 0:
        return "No listings match these filters."
    shown = min(total, limit)
    head = f"{total} matching listing(s)."
    if total > shown:
        head += f" Showing the first {shown}; tell the user there are {total - shown} more."
    lines = [head]
    for r in df.head(limit).itertuples():
        size = f"{r.area_marla:g} marla"
        if r.plot_count > 1:
            size += f" ({r.plot_count} plots)"
        notes = []
        if r.duplicate_count > 1 and r.price_max_pkr > r.price_min_pkr:
            notes.append(f"listed {r.duplicate_count}x, {cdl.price_label(r.price_min_pkr)}-{cdl.price_label(r.price_max_pkr)}")
        if r.flag_area_mismatch:
            notes.append("title size differs from listed area, verify")
        if r.flag_other_phase:
            notes.append(f"ad title says Phase {int(r.stated_phase)}, verify phase")
        if r.flag_area_adjusted:
            notes.append("total area calculated from per-plot size")
        line = (f"- Block {r.block_clean}, Plot {r.plot_clean} | {r.subType} | {size} | "
                f"{r.price_label} (PKR {r.price_pkr:,.0f}, {cdl.price_label(r.price_per_marla)}/marla)")
        if notes:
            line += " | NOTE: " + "; ".join(notes)
        lines.append(line)
    return "\n".join(lines)
