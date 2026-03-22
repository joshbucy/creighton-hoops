#!/usr/bin/env python3
"""
Creighton Bluejays Basketball Hub — Data Scraper
=================================================
Scrapes Warren Nolan for men's and women's team data,
generates AI summaries via the Claude API, and writes
everything to data.json for the React hub to consume.

Usage:
    python scraper.py

Requirements:
    pip install requests beautifulsoup4 anthropic

Schedule (cron example — runs daily at 6 AM):
    0 6 * * * /usr/bin/python3 /path/to/scraper.py

Environment variables:
    ANTHROPIC_API_KEY   — your Anthropic API key
    GITHUB_TOKEN        — (optional) for auto-committing data.json to GitHub
    GITHUB_REPO         — (optional) e.g. "yourusername/creighton-hoops"
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import anthropic
import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

MEN_URL         = "https://www.warrennolan.com/basketball/2026/schedule/Creighton"
BRACKET_MATRIX  = "http://www.bracketmatrix.com"
WOMEN_URL = "https://www.warrennolan.com/basketballw/2026/schedule/Creighton"
OUTPUT    = os.path.join(os.path.dirname(__file__), "data.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch(url: str) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object."""
    print(f"  Fetching {url}...")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(1)  # be polite
    return BeautifulSoup(resp.text, "html.parser")


def text(el) -> str:
    """Extract stripped text from a BS4 element, or '' if None."""
    return el.get_text(strip=True) if el else ""


def safe_int(s: str) -> int | None:
    """Parse an integer from a string, returning None on failure."""
    try:
        return int(re.sub(r"[^\d]", "", s))
    except (ValueError, TypeError):
        return None


# ── Scraper ───────────────────────────────────────────────────────────────────

def scrape_team(url: str) -> dict:
    """
    Scrape a Warren Nolan team schedule page and return a structured dict.
    Works for both MBB and WBB pages — same HTML structure.
    """
    soup = fetch(url)
    data = {}

    # ── Records block ──────────────────────────────────────────────────────
    # Warren Nolan puts records in a div with labeled cells
    record_cells = soup.select("div.record-item, div.records-block span, .team-records span")

    # Fallback: grab the header record string e.g. "Creighton Bluejays (15-17)"
    header = soup.find("h1") or soup.find("h2")
    if header:
        m = re.search(r"\((\d+)-(\d+)\)", text(header))
        if m:
            data["overall"] = f"{m.group(1)}-{m.group(2)}"

    # ── Rankings block ─────────────────────────────────────────────────────
    # Grab every labeled ranking row from the page
    rankings_raw = {}
    for row in soup.select("div.team-stats tr, table.rankings tr, .stat-block"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = text(cells[0]).strip().upper()
            val = text(cells[1]).strip()
            rankings_raw[key] = val

    # The rankings summary box uses specific CSS — grab it directly
    summary_box = soup.find(string=re.compile(r"NET\s*\d+", re.I))

    # More reliable: find the structured records/rankings section
    # Warren Nolan renders key stats in a consistent block
    def extract_labeled_value(label_pattern: str) -> str | None:
        """Find a value next to a label matching a regex pattern."""
        el = soup.find(string=re.compile(label_pattern, re.I))
        if el:
            parent = el.find_parent()
            if parent:
                # Value is often the next sibling element
                sibling = parent.find_next_sibling()
                if sibling:
                    return text(sibling)
                # Or a child
                children = [c for c in parent.children if hasattr(c, "get_text")]
                for child in children:
                    val = text(child).strip()
                    if val and val != text(el).strip():
                        return val
        return None

    # Parse the records section — Warren Nolan uses a consistent table
    records_section = soup.find("div", string=re.compile(r"Records", re.I))
    if not records_section:
        # Try finding it via structure
        for div in soup.find_all("div"):
            if "Overall" in text(div) and "Home" in text(div) and "Road" in text(div):
                records_section = div
                break

    # Parse all key-value pairs visible on the page using a broad sweep
    # This catches NET, RPI, ELO, SOS, quadrant records, etc.
    all_text = soup.get_text(separator="\n")
    lines = [l.strip() for l in all_text.splitlines() if l.strip()]

    def find_after(lines: list, label: str, offset: int = 1) -> str | None:
        """Find the value N lines after a label."""
        for i, line in enumerate(lines):
            if re.match(rf"^{label}$", line, re.I):
                target_idx = i + offset
                if target_idx < len(lines):
                    val = lines[target_idx].strip()
                    if val:
                        return val
        return None

    def find_record_after(lines: list, label: str) -> str | None:
        """Find a W-L record after a label."""
        val = find_after(lines, label)
        if val and re.match(r"^\d+-\d+$", val):
            return val
        return None

    # Extract records
    data["overall"]  = find_record_after(lines, "Overall")  or data.get("overall", "N/A")
    data["home"]     = find_record_after(lines, "Home")     or "N/A"
    data["road"]     = find_record_after(lines, "Road")     or "N/A"
    data["neutral"]  = find_record_after(lines, "Neutral")  or "N/A"
    data["conf"]     = find_record_after(lines, "Conf")     or "N/A"
    data["last10"]   = find_record_after(lines, "Last 10")  or "N/A"

    # Extract rankings
    data["net"] = find_after(lines, "NET")   or "N/A"
    data["rpi"] = find_after(lines, "RPI")   or "N/A"
    data["elo"] = find_after(lines, "ELO")   or "N/A"
    data["sos"] = find_after(lines, "SOS")   or "N/A"

    # Non-conf values often appear two lines after their label
    data["nonconf_rpi"] = find_after(lines, r"Non\s*[\n\r]*Conf", 2) or "N/A"
    data["nonconf_sos"] = find_after(lines, r"Non-Conference\s*[\n\r]*SOS", 2) or "N/A"

    # Streak
    data["streak"] = find_after(lines, "Streak") or "N/A"

    # Quadrant records — NET
    data["q1_net"] = find_record_after(lines, "Quadrant 1") or "N/A"
    data["q2_net"] = find_record_after(lines, "Quadrant 2") or "N/A"
    data["q3_net"] = find_record_after(lines, "Quadrant 3") or "N/A"
    data["q4_net"] = find_record_after(lines, "Quadrant 4") or "N/A"

    # ── Recent results ─────────────────────────────────────────────────────
    # Games are rendered as list items with date, opponent, result
    recent = []
    game_items = soup.select("ul > li, .schedule-game, .game-row")

    # Warren Nolan uses a specific structure — find scored games
    # Each game block contains date, opponent name, and a result like "W 76 - 59"
    game_blocks = soup.find_all(string=re.compile(r"^[WL]\s+\d+\s*-\s*\d+$"))

    for result_el in reversed(game_blocks):  # reversed = most recent last → we take last 5
        parent = result_el.find_parent()
        if not parent:
            continue

        # Walk up to the game container
        container = parent
        for _ in range(6):
            if container and container.find(string=re.compile(r"(JAN|FEB|MAR|NOV|DEC)", re.I)):
                break
            container = container.find_parent() if container else None

        if not container:
            continue

        result_text = result_el.strip()
        wl_match = re.match(r"([WL])\s+(\d+)\s*-\s*(\d+)", result_text)
        if not wl_match:
            continue

        wl    = wl_match.group(1)
        score1 = wl_match.group(2)
        score2 = wl_match.group(3)
        score  = f"{score1}–{score2}" if wl == "W" else f"{score2}–{score1}"

        # Date
        date_el = container.find(string=re.compile(
            r"(JAN|FEB|MAR|APR|NOV|DEC)\s*\d+", re.I
        ))
        date_str = text(date_el).strip() if date_el else "N/A"

        # Opponent — find a link that isn't Creighton
        opp_link = container.find("a", href=re.compile(r"/schedule/(?!Creighton)"))
        opp_name = text(opp_link) if opp_link else "Opponent"

        # Home/Away/Neutral context
        loc_el = container.find(string=re.compile(r"\b(AT|VS)\b", re.I))
        loc = text(loc_el).strip().upper() if loc_el else ""
        if loc == "AT":
            opp_display = f"at {opp_name}"
        elif loc == "VS":
            opp_display = f"vs {opp_name}"
        else:
            opp_display = opp_name

        recent.append({
            "date": date_str,
            "opponent": opp_display,
            "wl": wl,
            "score": score,
            "context": ""
        })

    # Keep only the 5 most recent completed games
    data["recent_results"] = recent[-5:] if len(recent) >= 5 else recent

    # ── Postseason info ────────────────────────────────────────────────────
    # Check for postseason record — Warren Nolan lists it
    data["postseason_record"] = find_record_after(lines, r"Post\s*[\n\r]*Season") or "0-0"

    return data


# ── AI Summary ────────────────────────────────────────────────────────────────

def generate_summary(team_data: dict, gender: str) -> str:
    """
    Call the Claude API to generate a 2-sentence season summary
    based on the scraped team data.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set — skipping AI summary.")
        return ""

    client = anthropic.Anthropic(api_key=api_key)

    recent_str = "\n".join(
        f"  {r['date']}: {r['wl']} {r['score']} {r['opponent']}"
        for r in team_data.get("recent_results", [])
    ) or "  No recent results available."

    prompt = f"""You are writing a brief season summary for the Creighton Bluejays {gender}'s basketball team for the 2025-26 season.

Here is their current data:
- Overall record: {team_data.get('overall', 'N/A')}
- Conference record: {team_data.get('conf', 'N/A')} (Big East)
- Home: {team_data.get('home', 'N/A')} | Away: {team_data.get('road', 'N/A')} | Neutral: {team_data.get('neutral', 'N/A')}
- NET ranking: {team_data.get('net', 'N/A')}
- RPI ranking: {team_data.get('rpi', 'N/A')}
- Last 10 games: {team_data.get('last10', 'N/A')}
- Current streak: {team_data.get('streak', 'N/A')}
- Quadrant 1 record: {team_data.get('q1_net', 'N/A')}
- Most recent 5 games:
{recent_str}

Write exactly 2 sentences summarizing where this team stands right now and their postseason outlook. Be specific and use the actual data above. Be direct — no preamble. Do not start with "Creighton" — vary the opening."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"  WARNING: Claude API error — {e}")
        return ""


# ── GitHub push (optional) ────────────────────────────────────────────────────

def push_to_github(filepath: str) -> None:
    """
    Optionally commit and push data.json to a GitHub repo.
    Requires GITHUB_TOKEN and GITHUB_REPO environment variables.
    Uses the GitHub Contents API (no git binary needed).
    """
    token = os.environ.get("GITHUB_TOKEN")
    repo  = os.environ.get("GITHUB_REPO")  # e.g. "yourusername/creighton-hoops"

    if not token or not repo:
        print("  Skipping GitHub push (GITHUB_TOKEN / GITHUB_REPO not set).")
        return

    import base64

    api_url = f"https://api.github.com/repos/{repo}/contents/data.json"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Read current file SHA (required for updates)
    sha = None
    resp = requests.get(api_url, headers=headers)
    if resp.status_code == 200:
        sha = resp.json().get("sha")

    with open(filepath, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "message": f"Auto-update data.json — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "content": content_b64,
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(api_url, headers=headers, json=payload)
    if resp.status_code in (200, 201):
        print("  ✓ data.json pushed to GitHub.")
    else:
        print(f"  ✗ GitHub push failed: {resp.status_code} {resp.text[:200]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_bracket_matrix(team: str = "Creighton") -> dict:
    """
    Scrape bracketmatrix.com for a team's bracket projection data.
    Only meaningful January through March during bracket season.
    Returns dict with keys: bm_status, bm_avg_seed, bm_bracket_count, bm_total_brackets
    """
    print(f"  Fetching Bracket Matrix for {team}...")
    try:
        soup = fetch(BRACKET_MATRIX)
        result = {
            "bm_status":         "N/A",
            "bm_avg_seed":       "N/A",
            "bm_bracket_count":  "N/A",
            "bm_total_brackets": "N/A",
        }

        # Find the table row for Creighton
        # Each team row contains the team name in a cell
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            row_text = text(cells[0])
            if team.lower() not in row_text.lower():
                continue

            all_cells = [text(c) for c in cells]

            # Total number of bracketologist columns (excluding header cols)
            # The matrix has ~5 fixed cols then one per bracketologist
            # Count non-empty seed values across the row to get bracket_count
            seed_values = []
            for val in all_cells[5:]:
                val = val.strip()
                if val and re.match(r"^\d{1,2}$", val):
                    seed_values.append(int(val))

            result["bm_bracket_count"]  = str(len(seed_values))
            result["bm_total_brackets"] = str(len(all_cells) - 5)

            # Average seed
            if seed_values:
                avg = sum(seed_values) / len(seed_values)
                result["bm_avg_seed"] = f"{avg:.1f}"
                result["bm_status"]   = "In"
            else:
                result["bm_avg_seed"] = "—"
                result["bm_status"]   = "Out"

            print(f"      → Status: {result['bm_status']}  "
                  f"Avg seed: {result['bm_avg_seed']}  "
                  f"Brackets: {result['bm_bracket_count']}/{result['bm_total_brackets']}")
            return result

        print(f"      → {team} not found in Bracket Matrix table")
        return result

    except Exception as e:
        print(f"      WARNING: Bracket Matrix scrape failed — {e}")
        return {
            "bm_status":         "N/A",
            "bm_avg_seed":       "N/A",
            "bm_bracket_count":  "N/A",
            "bm_total_brackets": "N/A",
        }


def main() -> None:
    print("=" * 60)
    print("Creighton Bluejays Basketball Hub — Scraper")
    print(f"Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    result = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "men": {},
        "women": {},
    }

    # ── Scrape men ─────────────────────────────────────────────────────────
    print("\n[1/4] Scraping men's data...")
    try:
        result["men"] = scrape_team(MEN_URL)
        print(f"      Overall: {result['men'].get('overall')}  NET: {result['men'].get('net')}")
    except Exception as e:
        print(f"      ERROR: {e}")
        result["men"] = {}

    # ── Scrape women ───────────────────────────────────────────────────────
    print("\n[2/4] Scraping women's data...")
    try:
        result["women"] = scrape_team(WOMEN_URL)
        print(f"      Overall: {result['women'].get('overall')}  NET: {result['women'].get('net')}")
    except Exception as e:
        print(f"      ERROR: {e}")
        result["women"] = {}

    # ── Scrape Bracket Matrix (January–March only) ────────────────────────
    current_month = datetime.now(timezone.utc).month
    if current_month in [1, 2, 3]:
        print("\n[2.5/4] Scraping Bracket Matrix (bracket season)...")
        bm_data = scrape_bracket_matrix("Creighton")
        result["men"].update(bm_data)
        print(f"      Status: {bm_data['bm_status']}  "
              f"Avg seed: {bm_data['bm_avg_seed']}  "
              f"In {bm_data['bm_bracket_count']} of {bm_data['bm_total_brackets']} brackets")
    else:
        print("\n[2.5/4] Bracket Matrix: skipping (not bracket season)")
        # Clear any stale bracket data from prior season
        for key in ["bm_status", "bm_avg_seed", "bm_bracket_count", "bm_total_brackets"]:
            result["men"][key] = "N/A"

    # ── Generate AI summaries ──────────────────────────────────────────────────
    print("\n[3/4] Generating AI summaries via Claude...")
    if result["men"]:
        print("      Generating men's summary...")
        result["men"]["summary"] = generate_summary(result["men"], "men")
        print(f"      → {result['men']['summary'][:80]}...")

    if result["women"]:
        print("      Generating women's summary...")
        result["women"]["summary"] = generate_summary(result["women"], "women")
        print(f"      → {result['women']['summary'][:80]}...")

    # ── Write data.json ────────────────────────────────────────────────────
    print(f"\n[4/4] Writing {OUTPUT}...")
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"      ✓ Written ({os.path.getsize(OUTPUT):,} bytes)")

    # ── Push to GitHub (optional) ──────────────────────────────────────────
    push_to_github(OUTPUT)

    print("\nDone! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
