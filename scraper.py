#!/usr/bin/env python3
"""
Creighton Bluejays Basketball Hub — Data Scraper
=================================================
Scrapes Warren Nolan for men's and women's team data,
generates AI summaries via the Claude API, and writes
everything to data.json for the hub to consume.

Usage:
    python scraper.py

Requirements:
    pip install requests beautifulsoup4 anthropic

Environment variables:
    ANTHROPIC_API_KEY   — your Anthropic API key
    GITHUB_TOKEN        — (optional) for auto-committing data.json to GitHub
    GITHUB_REPO         — (optional) e.g. "yourusername/creighton-hoops"
"""

import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import anthropic
import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

# >>> EACH NEW SEASON: change ONLY this number (season ending year) <<<
SEASON_END_YEAR     = 2027
SEASON_LABEL        = f"{SEASON_END_YEAR - 1}-{str(SEASON_END_YEAR)[2:]}"   # e.g. "2026-27"

MEN_SCHEDULE_URL    = f"https://www.warrennolan.com/basketball/{SEASON_END_YEAR}/schedule/Creighton"
MEN_SHEET_URL       = f"https://www.warrennolan.com/basketball/{SEASON_END_YEAR}/team-net-sheet?team=Creighton"
WOMEN_SCHEDULE_URL  = f"https://www.warrennolan.com/basketballw/{SEASON_END_YEAR}/schedule/Creighton"
WOMEN_SHEET_URL     = f"https://www.warrennolan.com/basketballw/{SEASON_END_YEAR}/team-net-sheet?team=Creighton"
# ESPN's public schedule feed (structured JSON) — used for the Last 5 Games table
ESPN_TEAM_ID        = "156"
ESPN_MEN_URL        = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/156/schedule"
ESPN_WOMEN_URL      = "https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/teams/156/schedule"
SEASON_START        = datetime(SEASON_END_YEAR - 1, 10, 1, tzinfo=timezone.utc)
try:
    CENTRAL = ZoneInfo("America/Chicago")
except Exception:
    CENTRAL = timezone(timedelta(hours=-6))

BRACKET_MATRIX      = "http://www.bracketmatrix.com"
OUTPUT              = os.path.join(os.path.dirname(__file__), "data.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def blank_team() -> dict:
    """Clean preseason state, used when the saved data belongs to a previous season."""
    return {
        "overall": "0-0", "home": "0-0", "road": "0-0", "neutral": "0-0",
        "conf": "0-0", "last10": "0-0", "streak": "N/A",
        "net": "N/A", "rpi": "N/A", "elo": "N/A", "sos": "N/A",
        "nonconf_rpi": "N/A", "nonconf_sos": "N/A",
        "q1_net": "0-0", "q2_net": "0-0", "q3_net": "0-0", "q4_net": "0-0",
        "ncaaTourney": "Out",
        "bm_status": "N/A", "bm_avg_seed": "N/A",
        "bm_bracket_count": "N/A", "bm_total_brackets": "N/A",
        "summary": f"Preseason — the {SEASON_LABEL} season hasn't tipped off yet. Check back once games begin.",
        "recent_results": [],
    }

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch(url: str) -> BeautifulSoup:
    print(f"  Fetching {url}...")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(1.5)
    return BeautifulSoup(resp.text, "html.parser")


def cel(el) -> str:
    """Safe text extraction from a BS4 element."""
    return el.get_text(strip=True) if el else ""


def is_rank(val: str) -> bool:
    """Returns True if val looks like a valid ranking number (1–400)."""
    try:
        n = int(val.strip())
        return 1 <= n <= 400
    except (ValueError, TypeError):
        return False


def is_record(val: str) -> bool:
    """Returns True if val looks like a W-L record."""
    return bool(re.match(r"^\d{1,3}-\d{1,3}$", (val or "").strip()))


def find_table_value(soup: BeautifulSoup, label: str) -> str:
    """
    Search all tables for a row where the first cell matches label
    and return the second cell's text. Case-insensitive.
    """
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                if label.lower() in cel(cells[0]).lower():
                    val = cel(cells[1]).strip()
                    if val:
                        return val
    return "N/A"


def find_labeled_value(soup: BeautifulSoup, label: str) -> str:
    """
    Find a text node matching label and return the text of its
    nearest following sibling or parent's next sibling.
    """
    el = soup.find(string=re.compile(rf"\b{re.escape(label)}\b", re.I))
    if not el:
        return "N/A"
    parent = el.find_parent()
    if not parent:
        return "N/A"
    # Try next sibling element
    sib = parent.find_next_sibling()
    if sib:
        val = cel(sib).strip()
        if val:
            return val
    # Try parent's next sibling
    grandparent = parent.find_parent()
    if grandparent:
        sib2 = parent.find_next_sibling()
        if sib2:
            val = cel(sib2).strip()
            if val:
                return val
    return "N/A"


def scrape_rankings(soup: BeautifulSoup) -> dict:
    """
    Extract NET, RPI, ELO, SOS rankings from a Warren Nolan page.
    Uses multiple strategies and validates that values are real rankings.
    """
    result = {"net": "N/A", "rpi": "N/A", "elo": "N/A", "sos": "N/A",
              "nonconf_rpi": "N/A", "nonconf_sos": "N/A", "streak": "N/A"}

    # Strategy 1: look for labeled <td> pairs in any table
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = cel(cells[0]).strip().upper()
            val   = cel(cells[1]).strip()
            if label == "NET"     and is_rank(val) and result["net"] == "N/A":
                result["net"] = val
            elif label == "RPI"   and is_rank(val) and result["rpi"] == "N/A":
                result["rpi"] = val
            elif label == "ELO"   and is_rank(val) and result["elo"] == "N/A":
                result["elo"] = val
            elif label == "SOS"   and is_rank(val) and result["sos"] == "N/A":
                result["sos"] = val
            elif "NON" in label and "RPI" in label and is_rank(val):
                result["nonconf_rpi"] = val
            elif "NON" in label and "SOS" in label and is_rank(val):
                result["nonconf_sos"] = val
            elif label == "STREAK" and val:
                result["streak"] = val

    # Strategy 2: look for divs/spans with specific class patterns
    # Warren Nolan sometimes uses stat boxes with class names like "net", "rpi" etc.
    for stat_key in ["net", "rpi", "elo", "sos"]:
        if result[stat_key] != "N/A":
            continue
        # Try finding elements with matching class or id
        for el in soup.find_all(class_=re.compile(stat_key, re.I)):
            val = cel(el).strip()
            # Often the label and value are both in the element; extract just the number
            nums = re.findall(r"\b(\d{1,3})\b", val)
            for n in nums:
                if is_rank(n):
                    result[stat_key] = n
                    break

    # Strategy 3: scan all text for labeled patterns like "NET: 83" or "NET 83"
    page_text = soup.get_text()
    for stat_key, pattern in [
        ("net", r"\bNET[:\s]+(\d{1,3})\b"),
        ("rpi", r"\bRPI[:\s]+(\d{1,3})\b"),
        ("elo", r"\bELO[:\s]+(\d{1,3})\b"),
        ("sos", r"\bSOS[:\s]+(\d{1,3})\b"),
    ]:
        if result[stat_key] == "N/A":
            m = re.search(pattern, page_text, re.I)
            if m and is_rank(m.group(1)):
                result[stat_key] = m.group(1)

    return result


def scrape_records(soup: BeautifulSoup) -> dict:
    """
    Extract W-L records (overall, home, road, neutral, conf, last10)
    from a Warren Nolan page using multiple strategies.
    """
    result = {
        "overall": "N/A", "home": "N/A", "road": "N/A",
        "neutral": "N/A", "conf":  "N/A", "last10": "N/A"
    }

    label_map = {
        "overall": ["overall", "all games"],
        "home":    ["home"],
        "road":    ["road", "away"],
        "neutral": ["neutral"],
        "conf":    ["conf", "conference"],
        "last10":  ["last 10", "last10"],
    }

    # Strategy 1: table rows
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = cel(cells[0]).strip().lower()
            val   = cel(cells[1]).strip()
            for key, patterns in label_map.items():
                if result[key] == "N/A":
                    for pat in patterns:
                        if pat in label and is_record(val):
                            result[key] = val
                            break

    # Strategy 2: regex scan of full page text
    page_text = soup.get_text()
    record_patterns = {
        "overall": r"Overall[:\s]+(\d+-\d+)",
        "home":    r"Home[:\s]+(\d+-\d+)",
        "road":    r"(?:Road|Away)[:\s]+(\d+-\d+)",
        "neutral": r"Neutral[:\s]+(\d+-\d+)",
        "conf":    r"(?:Conf|Conference)[:\s]+(\d+-\d+)",
        "last10":  r"Last\s*10[:\s]+(\d+-\d+)",
    }
    for key, pattern in record_patterns.items():
        if result[key] == "N/A":
            m = re.search(pattern, page_text, re.I)
            if m and is_record(m.group(1)):
                result[key] = m.group(1)

    return result


def scrape_quadrants(soup: BeautifulSoup) -> dict:
    """Extract Q1–Q4 records from a Warren Nolan page."""
    result = {"q1_net": "N/A", "q2_net": "N/A", "q3_net": "N/A", "q4_net": "N/A"}

    page_text = soup.get_text()
    for key, pattern in [
        ("q1_net", r"Quadrant\s*1[:\s]+(\d+-\d+)"),
        ("q2_net", r"Quadrant\s*2[:\s]+(\d+-\d+)"),
        ("q3_net", r"Quadrant\s*3[:\s]+(\d+-\d+)"),
        ("q4_net", r"Quadrant\s*4[:\s]+(\d+-\d+)"),
    ]:
        m = re.search(pattern, page_text, re.I)
        if m and is_record(m.group(1)):
            result[key] = m.group(1)

    # Also try table rows
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = cel(cells[0]).strip()
            val   = cel(cells[1]).strip()
            for i, key in enumerate(["q1_net","q2_net","q3_net","q4_net"], 1):
                if result[key] == "N/A" and re.search(rf"quadrant\s*{i}", label, re.I):
                    if is_record(val):
                        result[key] = val

    return result


def scrape_recent_results(soup: BeautifulSoup) -> list:
    """
    Extract the 5 most recent completed games from a Warren Nolan schedule page.
    """
    results = []

    # Warren Nolan marks results with W or L followed by scores
    # Look for table rows containing a result
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            row_text = " ".join(cel(c) for c in cells)
            # Must contain a W or L result pattern
            result_match = re.search(r"\b([WL])\s+(\d+)\s*[-–]\s*(\d+)\b", row_text)
            if not result_match:
                continue

            wl     = result_match.group(1)
            score1 = result_match.group(2)
            score2 = result_match.group(3)

            # Score: winner's points first
            if wl == "W":
                score = f"{score1}-{score2}"
            else:
                score = f"{score2}-{score1}"

            # Date — look for month pattern
            date_match = re.search(r"(Jan|Feb|Mar|Apr|Nov|Dec)\s+\d+", row_text, re.I)
            date_str = date_match.group(0) if date_match else "N/A"

            # Opponent — any cell that isn't the result or date
            opponent = "N/A"
            for c in cells:
                ct = cel(c).strip()
                if ct and not re.search(r"^[WL]\s+\d", ct) and not re.search(r"(Jan|Feb|Mar|Nov|Dec)\s+\d", ct, re.I) and len(ct) > 2:
                    opponent = ct
                    break

            # Context — look for tournament/neutral indicators
            context = ""
            if re.search(r"tourna|tournament|neutral|@", row_text, re.I):
                context = "Tournament" if re.search(r"tourna", row_text, re.I) else "Neutral"

            results.append({
                "date":     date_str,
                "opponent": opponent,
                "wl":       wl,
                "score":    score,
                "context":  context
            })

    # Return 5 most recent (last in list = most recent)
    print(f"  Warren Nolan result rows parsed: {len(results)}")
    return list(reversed(results[-5:]))   # newest first


def validate_and_merge(scraped: dict, previous: dict) -> dict:
    """
    Validate scraped data and fall back to previous values for any
    fields that look wrong. This prevents bad scrapes from nuking good data.
    """
    result = dict(previous)  # start with previous as base

    # Rankings must be numeric strings 1–400
    for key in ["net", "rpi", "elo", "sos", "nonconf_rpi", "nonconf_sos"]:
        val = scraped.get(key, "N/A")
        if is_rank(val):
            result[key] = val
        # else keep previous

    # Records must match W-L format
    for key in ["overall", "home", "road", "neutral", "conf", "last10",
                "q1_net", "q2_net", "q3_net", "q4_net"]:
        val = scraped.get(key, "N/A")
        if is_record(val):
            result[key] = val
        # else keep previous

    # Streak — any non-empty string is fine
    if scraped.get("streak") and scraped["streak"] != "N/A":
        result["streak"] = scraped["streak"]

    # Recent results — keep every game that passes validation (even just one,
    # early in the season); bad rows are dropped individually.
    good = [r for r in scraped.get("recent_results", []) if valid_result(r)]
    if good:
        result["recent_results"] = good

    # Summary — always update if present
    if scraped.get("summary"):
        result["summary"] = scraped["summary"]

    # Bracket matrix fields — always update
    for key in ["bm_status", "bm_avg_seed", "bm_bracket_count", "bm_total_brackets"]:
        if key in scraped:
            result[key] = scraped[key]

    # ncaaTourney — always update
    if "ncaaTourney" in scraped:
        result["ncaaTourney"] = scraped["ncaaTourney"]

    return result


def scrape_team(schedule_url: str, sheet_url: str) -> dict:
    """
    Scrape a Warren Nolan team page and return a structured dict.
    Fetches both the schedule page and the team sheet for redundancy.
    """
    data = {}

    # Fetch schedule page (has recent results + records)
    schedule_soup = fetch(schedule_url)
    data.update(scrape_records(schedule_soup))
    data.update(scrape_rankings(schedule_soup))
    data.update(scrape_quadrants(schedule_soup))
    data["recent_results"] = scrape_recent_results(schedule_soup)

    # Fetch team sheet page (has cleaner rankings + quadrant data)
    try:
        sheet_soup = fetch(sheet_url)
        sheet_rankings  = scrape_rankings(sheet_soup)
        sheet_records   = scrape_records(sheet_soup)
        sheet_quadrants = scrape_quadrants(sheet_soup)

        # Prefer sheet data for rankings (it's more reliable there)
        for key in ["net", "rpi", "elo", "sos", "nonconf_rpi", "nonconf_sos"]:
            if is_rank(sheet_rankings.get(key, "N/A")):
                data[key] = sheet_rankings[key]

        # Prefer sheet data for quadrants
        for key in ["q1_net", "q2_net", "q3_net", "q4_net"]:
            if is_record(sheet_quadrants.get(key, "N/A")):
                data[key] = sheet_quadrants[key]

        # Fill in any still-missing records from sheet
        for key in ["overall", "home", "road", "neutral", "conf", "last10"]:
            if data.get(key) == "N/A" and is_record(sheet_records.get(key, "N/A")):
                data[key] = sheet_records[key]

    except Exception as e:
        print(f"  WARNING: Team sheet fetch failed — {e}")

    missing = [k for k in ("overall", "home", "road", "neutral", "conf", "last10",
                           "net", "rpi", "elo", "sos", "q1_net", "streak")
               if data.get(k) in (None, "", "N/A")]
    print(f"  Warren Nolan fields not found: {missing or 'none'}")
    return data


# ── Result validation, ESPN schedule, sanity checks ──────────────────────────

def valid_result(r: dict) -> bool:
    """A game row is usable only if every field looks like a real game."""
    return (
        isinstance(r, dict)
        and r.get("wl") in ("W", "L")
        and bool(re.match(r"^\d+-\d+$", str(r.get("score", ""))))
        and r.get("date") not in (None, "", "N/A")
        and r.get("opponent") not in (None, "", "N/A")
    )


def total_games(rec) -> int | None:
    m = re.match(r"^(\d+)-(\d+)$", str(rec or "").strip())
    return int(m.group(1)) + int(m.group(2)) if m else None


def _parse_espn_date(s):
    for fmt in ("%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _espn_score(c: dict):
    sc = c.get("score")
    if isinstance(sc, dict):
        sc = sc.get("displayValue", sc.get("value"))
    try:
        return int(float(sc))
    except (TypeError, ValueError):
        return None


def fetch_espn_games(url: str) -> list:
    """Completed, non-exhibition games this season, oldest first."""
    print(f"  Fetching {url}...")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    games = []
    for ev in resp.json().get("events", []):
        try:
            comp = (ev.get("competitions") or [{}])[0]
            status = ((comp.get("status") or ev.get("status") or {}).get("type")) or {}
            if not status.get("completed"):
                continue
            if str((ev.get("seasonType") or {}).get("type")) == "1":   # preseason/exhibition
                continue
            when = _parse_espn_date(ev.get("date"))
            if not when or when < SEASON_START:
                continue
            us = opp = None
            for c in comp.get("competitors", []):
                cid = str((c.get("team") or {}).get("id", c.get("id")))
                if cid == ESPN_TEAM_ID:
                    us = c
                else:
                    opp = c
            if not us or not opp:
                continue
            pf, pa = _espn_score(us), _espn_score(opp)
            if pf is None or pa is None or pf == pa:
                continue
            team = opp.get("team") or {}
            name = team.get("location") or team.get("shortDisplayName") or team.get("displayName") or "Opponent"
            rank = (opp.get("curatedRank") or {}).get("current")
            if isinstance(rank, int) and 1 <= rank <= 25:
                name = f"#{rank} {name}"
            where = "neutral" if comp.get("neutralSite") else ("away" if us.get("homeAway") == "away" else "home")
            notes = comp.get("notes") or []
            context = (notes[0].get("headline", "") if notes and isinstance(notes[0], dict) else "")
            games.append({
                "when": when.astimezone(CENTRAL), "win": pf > pa, "pf": pf, "pa": pa,
                "opp": name, "where": where,
                "conf": bool(comp.get("conferenceCompetition")), "context": context[:40],
            })
        except Exception as e:                      # one odd event shouldn't sink the rest
            print(f"  ESPN: skipped an event ({type(e).__name__}: {e})")
    games.sort(key=lambda g: g["when"])
    return games


def games_to_recent(games: list) -> list:
    out = []
    for g in reversed(games[-5:]):                  # newest first
        out.append({
            "date": f"{g['when'].strftime('%b')} {g['when'].day}",
            "opponent": ("at " if g["where"] == "away" else "vs ") + g["opp"],
            "wl": "W" if g["win"] else "L",
            "score": f"{max(g['pf'], g['pa'])}-{min(g['pf'], g['pa'])}",
            "context": g["context"] or ("Conference" if g["conf"] else ""),
        })
    return out


def games_to_records(games: list) -> dict:
    def rec(sub):
        w = sum(1 for g in sub if g["win"])
        return f"{w}-{len(sub) - w}"
    last = games[-1]["win"]
    n = 0
    for g in reversed(games):
        if g["win"] != last:
            break
        n += 1
    return {
        "overall": rec(games),
        "home":    rec([g for g in games if g["where"] == "home"]),
        "road":    rec([g for g in games if g["where"] == "away"]),
        "neutral": rec([g for g in games if g["where"] == "neutral"]),
        "conf":    rec([g for g in games if g["conf"]]),
        "last10":  rec(games[-10:]),
        "streak":  f"{n}{'W' if last else 'L'}",
    }


def espn_overlay(team: dict, url: str, fill_records: bool) -> dict:
    """Last 5 results from ESPN; also W-L splits if Warren Nolan was unavailable."""
    try:
        games = fetch_espn_games(url)
    except Exception as e:
        print(f"  ESPN unavailable ({type(e).__name__}: {e}) — leaving results as they are")
        return team
    if not games:
        print("  ESPN: no completed games found for this season yet")
        return team
    team = dict(team)
    team["recent_results"] = games_to_recent(games)
    print(f"  ESPN: {len(games)} completed game(s); showing the latest {len(team['recent_results'])}")
    if fill_records:
        team.update(games_to_records(games))
        print(f"  ESPN: filled records (Warren Nolan unavailable) -> overall {team['overall']}")
    return team


def finalize_team(t: dict) -> dict:
    """Final sanity pass so impossible combinations never reach the site."""
    t = dict(t)
    played = total_games(t.get("overall"))
    if played is not None and played <= 10:
        t["last10"] = t.get("overall")              # under 10 games: last 10 == season record
    rr = t.get("recent_results") or []
    if t.get("streak") in (None, "", "N/A") and rr:
        first, n = rr[0]["wl"], 0
        for r in rr:
            if r["wl"] != first:
                break
            n += 1
        t["streak"] = f"{n}{first}" + ("+" if n == len(rr) >= 5 else "")
    return t


# ── AI Summary ────────────────────────────────────────────────────────────────

def generate_summary(team_data: dict, gender: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set — skipping AI summary.")
        return ""

    client = anthropic.Anthropic(api_key=api_key)

    recent_str = "\n".join(
        f"  {r['date']}: {r['wl']} {r['score']} {r['opponent']}"
        for r in team_data.get("recent_results", [])
    ) or "  No recent results available."

    is_preseason = team_data.get("overall", "0-0") in ("0-0", "N/A", "")
    phase_note = (
        "The season has NOT started yet (0-0 record, no games played). Write a preseason "
        "outlook that builds anticipation without inventing specific facts about the roster, "
        "recruits, or rankings. Do not cite stats that say N/A."
        if is_preseason else
        "The season is in progress or complete; summarize using the actual data below."
    )

    prompt = f"""You are writing a brief season summary for the Creighton Bluejays {gender}'s basketball team for the {SEASON_LABEL} season.

{phase_note}

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
            model="claude-sonnet-5-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        # Newer models may return a thinking block before the answer, so
        # collect only the text blocks instead of assuming content[0] is text.
        text_out = "".join(
            b.text for b in message.content if getattr(b, "type", "") == "text"
        ).strip()
        if not text_out:
            print("  WARNING: Claude returned no text content")
        return text_out
    except Exception as e:
        print(f"  WARNING: Claude API error ({type(e).__name__}) — {e}")
        return ""


# ── Bracket Matrix ────────────────────────────────────────────────────────────

def scrape_bracket_matrix(team: str = "Creighton") -> dict:
    """
    Scrape bracketmatrix.com for a team's bracket projection data.
    Only meaningful January through March during bracket season.
    """
    empty = {
        "bm_status": "N/A", "bm_avg_seed": "N/A",
        "bm_bracket_count": "N/A", "bm_total_brackets": "N/A"
    }
    print(f"  Fetching Bracket Matrix for {team}...")
    try:
        soup = fetch(BRACKET_MATRIX)
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            if team.lower() not in cel(cells[0]).lower():
                continue
            all_cells = [cel(c) for c in cells]
            seed_values = []
            for val in all_cells[5:]:
                val = val.strip()
                if val and re.match(r"^\d{1,2}$", val):
                    seed_values.append(int(val))
            count = len(seed_values)
            total = len(all_cells) - 5
            if seed_values:
                avg = sum(seed_values) / len(seed_values)
                result = {
                    "bm_status":         "In",
                    "bm_avg_seed":       f"{avg:.1f}",
                    "bm_bracket_count":  str(count),
                    "bm_total_brackets": str(total),
                }
            else:
                result = {
                    "bm_status":         "Out",
                    "bm_avg_seed":       "—",
                    "bm_bracket_count":  "0",
                    "bm_total_brackets": str(total),
                }
            print(f"      → {result['bm_status']}  Avg seed: {result['bm_avg_seed']}  "
                  f"Brackets: {result['bm_bracket_count']}/{result['bm_total_brackets']}")
            return result

        print(f"      → {team} not found in Bracket Matrix")
        return empty
    except Exception as e:
        print(f"      WARNING: Bracket Matrix scrape failed — {e}")
        return empty


# ── GitHub push (optional) ────────────────────────────────────────────────────

def push_to_github(filepath: str) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo  = os.environ.get("GITHUB_REPO")
    if not token or not repo:
        print("  Skipping GitHub push (GITHUB_TOKEN / GITHUB_REPO not set).")
        return
    import base64
    api_url = f"https://api.github.com/repos/{repo}/contents/data.json"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
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

def main() -> None:
    print("=" * 60)
    print("Creighton Bluejays Basketball Hub — Scraper")
    print(f"Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Load previous data.json to use as fallback for failed scrapes
    previous_men   = blank_team()
    previous_women = blank_team()
    if os.path.exists(OUTPUT):
        try:
            with open(OUTPUT) as f:
                prev = json.load(f)
            if prev.get("season") == SEASON_LABEL:
                previous_men   = prev.get("men",   {})
                previous_women = prev.get("women", {})
                print(f"\nLoaded previous data.json as fallback.")
            else:
                print(f"\nSaved data is from a different season "
                      f"({prev.get('season', 'unlabeled')}) — starting {SEASON_LABEL} fresh.")
        except Exception:
            pass

    result = {
        "season": SEASON_LABEL,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "men":   {},
        "women": {},
    }

    # ── Scrape men ─────────────────────────────────────────────────────────
    print("\n[1/5] Scraping men's data...")
    men_wn_failed = False
    try:
        raw_men = scrape_team(MEN_SCHEDULE_URL, MEN_SHEET_URL)
        result["men"] = validate_and_merge(raw_men, previous_men)
        print(f"      Overall: {result['men'].get('overall')}  "
              f"NET: {result['men'].get('net')}  "
              f"Games: {len(result['men'].get('recent_results', []))}")
    except Exception as e:
        print(f"      Not available yet or failed ({e}) — keeping previous/blank data")
        result["men"] = previous_men
        men_wn_failed = True

    # ── Scrape women ───────────────────────────────────────────────────────
    print("\n[2/5] Scraping women's data...")
    women_wn_failed = False
    try:
        raw_women = scrape_team(WOMEN_SCHEDULE_URL, WOMEN_SHEET_URL)
        result["women"] = validate_and_merge(raw_women, previous_women)
        print(f"      Overall: {result['women'].get('overall')}  "
              f"NET: {result['women'].get('net')}  "
              f"Games: {len(result['women'].get('recent_results', []))}")
    except Exception as e:
        print(f"      Not available yet or failed ({e}) — keeping previous/blank data")
        result["women"] = previous_women
        women_wn_failed = True

    # ── ESPN schedule feed: Last 5 results, plus records if Warren Nolan failed ──
    print("\n[2.5/5] Recent results from ESPN...")
    result["men"]   = finalize_team(espn_overlay(result["men"],   ESPN_MEN_URL,   men_wn_failed))
    result["women"] = finalize_team(espn_overlay(result["women"], ESPN_WOMEN_URL, women_wn_failed))

    # ── Scrape Bracket Matrix (January–March only) ─────────────────────────
    current_month = datetime.now(timezone.utc).month
    print(f"\n[3/5] Bracket Matrix...")
    if current_month in [1, 2, 3]:
        bm = scrape_bracket_matrix("Creighton")
        result["men"].update(bm)
    else:
        print("  Skipping — not bracket season (Jan–Mar only)")
        for key in ["bm_status", "bm_avg_seed", "bm_bracket_count", "bm_total_brackets"]:
            result["men"][key] = "N/A"

    # ── Generate AI summaries ──────────────────────────────────────────────
    print("\n[4/5] Generating AI summaries via Claude...")
    if result["men"]:
        print("  Generating men's summary...")
        new_summary = generate_summary(result["men"], "men")
        result["men"]["summary"] = new_summary or previous_men.get("summary", "")
        if new_summary:
            print(f"  → {result['men']['summary'][:80]}...")

    if result["women"]:
        print("  Generating women's summary...")
        new_summary = generate_summary(result["women"], "women")
        result["women"]["summary"] = new_summary or previous_women.get("summary", "")
        if new_summary:
            print(f"  → {result['women']['summary'][:80]}...")

    # ── Write data.json ────────────────────────────────────────────────────
    print(f"\n[5/5] Writing {OUTPUT}...")
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"      ✓ Written ({os.path.getsize(OUTPUT):,} bytes)")

    # ── Push to GitHub (optional) ──────────────────────────────────────────
    push_to_github(OUTPUT)

    print("\nDone! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
