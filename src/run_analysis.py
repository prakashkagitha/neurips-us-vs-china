"""
Pipeline for analyzing NeurIPS accepted papers with a focus on US vs China research output.

This script:
1. Normalizes author affiliation strings across multiple NeurIPS datasets.
2. Maps normalized affiliations to countries and consolidated regions.
3. Produces summary tables and visualizations that highlight regional trends.

Outputs are written to the usvschina directory:
- affiliation_normalization.json
- affiliation_region_mapping.json
- region_paper_counts.csv
- top_affiliations_by_year.csv
- us_china_collaboration.csv
- figures/region_trends.png
- figures/top_affiliation_bump.png
- figures/us_china_collaboration.png
- figures/region_share.png
- figures/region_share_stacked.png
- affiliation_counts_full.csv
- unknown_region_papers.csv
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PAPERTRAILS_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Region definitions
# --------------------------------------------------------------------------------------

REGION_COUNTRIES: Dict[str, Set[str]] = {
    "United States": {"United States"},
    "China": {"China", "Mainland China", "Hong Kong", "Hong Kong SAR", "Hong Kong SAR China", "Macau", "Taiwan"},
    "Canada": {"Canada"},
    "Europe": {
        "Austria",
        "Belgium",
        "Croatia",
        "Czech Republic",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Iceland",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Netherlands",
        "Norway",
        "Poland",
        "Portugal",
        "Romania",
        "Russia",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
        "United Kingdom",
        "UK",
    },
    "Asia-Pacific (excl. China)": {
        "Australia",
        "Bangladesh",
        "Bhutan",
        "Brunei",
        "Cambodia",
        "Indonesia",
        "India",
        "Japan",
        "Laos",
        "Malaysia",
        "Mongolia",
        "Myanmar",
        "Nepal",
        "New Zealand",
        "Pakistan",
        "Philippines",
        "Singapore",
        "South Korea",
        "Sri Lanka",
        "Thailand",
        "Vietnam",
    },
    "Middle East & Africa": {
        "Algeria",
        "Bahrain",
        "Botswana",
        "Cameroon",
        "Côte d’Ivoire",
        "Egypt",
        "Ethiopia",
        "Ghana",
        "Iran",
        "Iraq",
        "Israel",
        "Jordan",
        "Kenya",
        "Kuwait",
        "Lebanon",
        "Morocco",
        "Namibia",
        "Nigeria",
        "Oman",
        "Qatar",
        "Saudi Arabia",
        "South Africa",
        "Tanzania",
        "Tunisia",
        "Turkey",
        "Uganda",
        "United Arab Emirates",
    },
    "Latin America": {
        "Argentina",
        "Bolivia",
        "Brazil",
        "Chile",
        "Colombia",
        "Costa Rica",
        "Cuba",
        "Dominican Republic",
        "Ecuador",
        "El Salvador",
        "Guatemala",
        "Honduras",
        "Mexico",
        "Panama",
        "Paraguay",
        "Peru",
        "Puerto Rico",
        "Uruguay",
        "Venezuela",
    },
    "Other/Unknown": set(),
}

REGION_ORDER: List[str] = [
    "United States",
    "China",
    "Asia-Pacific (excl. China)",
    "Europe",
    "Canada",
    "Middle East & Africa",
    "Latin America",
]


def allocate_label_position(existing: List[float], desired: float, spacing: float = 10.0) -> Tuple[float, bool]:
    """Return a label position and whether an arrow is needed to point back to the data point."""

    if not existing:
        existing.append(desired)
        return desired, False

    offsets = [0.0]
    step = spacing
    # Generate alternating offsets: +step, -step, +2*step, -2*step, ...
    for k in range(1, 8):
        offsets.append(step * k)
        offsets.append(-step * k)

    for offset in offsets:
        candidate = desired + offset
        if all(abs(candidate - pos) >= spacing for pos in existing):
            existing.append(candidate)
            return candidate, abs(offset) > 1e-6

    # Fallback: push far enough away
    candidate = desired + spacing * (len(existing) + 1)
    existing.append(candidate)
    return candidate, True


COUNTRY_SYNONYMS: Dict[str, str] = {
    "usa": "United States",
    "u s a": "United States",
    "u s": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "pr china": "China",
    "people s republic of china": "China",
    "peoples republic of china": "China",
    "mainland china": "China",
    "p r china": "China",
    "hong kong sar china": "Hong Kong",
    "hong kong sar": "Hong Kong",
    "hong kong special administrative region": "Hong Kong",
    "hksar": "Hong Kong",
    "taiwan roc": "Taiwan",
    "taiwan r o c": "Taiwan",
    "taiwan (roc)": "Taiwan",
    "korea republic of": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "korea": "South Korea",
    "s korea": "South Korea",
    "u k": "United Kingdom",
    "uk": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "new york ny": "United States",
    "montreal qc": "Canada",
    "toronto on": "Canada",
    "vancouver bc": "Canada",
    "sydney nsw": "Australia",
    "melbourne vic": "Australia",
    "beijing": "China",
    "shanghai": "China",
    "shenzhen": "China",
    "hangzhou": "China",
    "guangzhou": "China",
    "nanjing": "China",
    "wuhan": "China",
    "tokyo": "Japan",
    "kyoto": "Japan",
    "singapore": "Singapore",
    "lausanne": "Switzerland",
    "zurich": "Switzerland",
    "paris": "France",
    "berlin": "Germany",
    "munich": "Germany",
    "munchen": "Germany",
    "cambridge uk": "United Kingdom",
    "cambridge united kingdom": "United Kingdom",
    "cambridge ma": "United States",
}


def _normalize_text(text: str) -> str:
    """Lowercase, ASCII-safe normalization for matching."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _title_case(text: str) -> str:
    """Converts a normalized string back to a readable title case."""
    if not text:
        return ""
    return " ".join(word.capitalize() if word else "" for word in text.split())


# --------------------------------------------------------------------------------------
# Manual normalization rules
# --------------------------------------------------------------------------------------

ManualRule = Tuple[str, str, str]


MANUAL_EXACT_RAW: Dict[str, ManualRule] = {
    "cmu": ("Carnegie Mellon University", "United States", "manual_exact"),
    "mit": ("Massachusetts Institute of Technology", "United States", "manual_exact"),
    "caltech": ("California Institute of Technology", "United States", "manual_exact"),
    "uc berkeley": ("University of California, Berkeley", "United States", "manual_exact"),
    "ucb": ("University of California, Berkeley", "United States", "manual_exact"),
    "ucla": ("University of California, Los Angeles", "United States", "manual_exact"),
    "ucsd": ("University of California, San Diego", "United States", "manual_exact"),
    "uc san diego": ("University of California, San Diego", "United States", "manual_exact"),
    "uc davis": ("University of California, Davis", "United States", "manual_exact"),
    "uc irvine": ("University of California, Irvine", "United States", "manual_exact"),
    "uc santa barbara": ("University of California, Santa Barbara", "United States", "manual_exact"),
    "uc santa cruz": ("University of California, Santa Cruz", "United States", "manual_exact"),
    "uiuc": ("University of Illinois Urbana-Champaign", "United States", "manual_exact"),
    "uiuc urbana champaign": ("University of Illinois Urbana-Champaign", "United States", "manual_exact"),
    "umich": ("University of Michigan - Ann Arbor", "United States", "manual_exact"),
    "gatech": ("Georgia Institute of Technology", "United States", "manual_exact"),
    "georgia tech": ("Georgia Institute of Technology", "United States", "manual_exact"),
    "uw": ("University of Washington", "United States", "manual_exact"),
    "uw seattle": ("University of Washington", "United States", "manual_exact"),
    "utaustin": ("University of Texas at Austin", "United States", "manual_exact"),
    "ut austin": ("University of Texas at Austin", "United States", "manual_exact"),
    "ucf": ("University of Central Florida", "United States", "manual_exact"),
    "nus": ("National University of Singapore", "Singapore", "manual_exact"),
    "ntu": ("Nanyang Technological University", "Singapore", "manual_exact"),
    "hkust": ("Hong Kong University of Science and Technology", "Hong Kong", "manual_exact"),
    "hku": ("University of Hong Kong", "Hong Kong", "manual_exact"),
    "cuhk": ("The Chinese University of Hong Kong", "Hong Kong", "manual_exact"),
    "ustc": ("University of Science and Technology of China", "China", "manual_exact"),
    "tsinghua": ("Tsinghua University", "China", "manual_exact"),
    "zju": ("Zhejiang University", "China", "manual_exact"),
    "sjtu": ("Shanghai Jiao Tong University", "China", "manual_exact"),
    "nju": ("Nanjing University", "China", "manual_exact"),
    "pku": ("Peking University", "China", "manual_exact"),
    "hit": ("Harbin Institute of Technology", "China", "manual_exact"),
    "sustech": ("Southern University of Science and Technology", "China", "manual_exact"),
    "ust": ("Hong Kong University of Science and Technology", "Hong Kong", "manual_exact"),
    "epfl": ("École Polytechnique Fédérale de Lausanne", "Switzerland", "manual_exact"),
    "eth": ("ETH Zurich", "Switzerland", "manual_exact"),
    "ethz": ("ETH Zurich", "Switzerland", "manual_exact"),
    "inria": ("Inria", "France", "manual_exact"),
    "ens": ("École Normale Supérieure", "France", "manual_exact"),
    "ubc": ("University of British Columbia", "Canada", "manual_exact"),
    "utoronto": ("University of Toronto", "Canada", "manual_exact"),
    "u of t": ("University of Toronto", "Canada", "manual_exact"),
    "u waterloo": ("University of Waterloo", "Canada", "manual_exact"),
    "uwaterloo": ("University of Waterloo", "Canada", "manual_exact"),
    "ubolder": ("University of Colorado Boulder", "United States", "manual_exact"),
    "sfu": ("Simon Fraser University", "Canada", "manual_exact"),
    "national yang ming chiao tung university": ("National Yang Ming Chiao Tung University", "Taiwan", "manual_exact"),
    "national chiao tung university": ("National Yang Ming Chiao Tung University", "Taiwan", "manual_exact"),
    "national tsinghua university": ("National Tsing Hua University", "Taiwan", "manual_exact"),
    "national tsing hua university": ("National Tsing Hua University", "Taiwan", "manual_exact"),
    "city university of macau": ("City University of Macau", "China", "manual_exact"),
    "university of macau": ("University of Macau", "China", "manual_exact"),
    "incheon national university": ("Incheon National University", "South Korea", "manual_exact"),
    "sungkyunkwan university": ("Sungkyunkwan University", "South Korea", "manual_exact"),
    "ewha womans university": ("Ewha Womans University", "South Korea", "manual_exact"),
    "chung ang university": ("Chung-Ang University", "South Korea", "manual_exact"),
    "sookmyung women s university": ("Sookmyung Women's University", "South Korea", "manual_exact"),
    "sookmyung women university": ("Sookmyung Women's University", "South Korea", "manual_exact"),
    "middle east technical university": ("Middle East Technical University", "Turkey", "manual_exact"),
    "bogazici university": ("Boğaziçi University", "Turkey", "manual_exact"),
    "lomonosov moscow state university": ("Lomonosov Moscow State University", "Russia", "manual_exact"),
    "higher school of economics": ("National Research University Higher School of Economics", "Russia", "manual_exact"),
    "sung kyun kwan university": ("Sungkyunkwan University", "South Korea", "manual_exact"),
    "itmo university": ("ITMO University", "Russia", "manual_exact"),
}

MANUAL_EXACT: Dict[str, ManualRule] = {
    _normalize_text(key): value for key, value in MANUAL_EXACT_RAW.items()
}


RAW_MANUAL_SUBSTRINGS: List[ManualRule] = [
    ("mohamed bin zayed university of ai", "Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)", "United Arab Emirates"),
    ("mbzuai", "Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)", "United Arab Emirates"),
    ("mohamed bin zayed univeristy of ai", "Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)", "United Arab Emirates"),
    ("cerebras systems", "Cerebras Systems", "United States"),
    ("boson ai", "Boson AI", "United States"),
    ("atomwise", "Atomwise", "United States"),
    ("asapp", "ASAPP", "United States"),
    ("sambanova", "SambaNova Systems", "United States"),
    ("mathworks", "MathWorks", "United States"),
    ("multion", "Multion", "United States"),
    ("flatiron institute", "Flatiron Institute", "United States"),
    ("optimizely", "Optimizely", "United States"),
    ("linkedin", "LinkedIn", "United States"),
    ("webank", "WeBank", "China"),
    ("we bank", "WeBank", "China"),
    ("mediatek", "MediaTek", "Taiwan"),
    ("goldman sachs", "Goldman Sachs", "United States"),
    ("sangfor", "Sangfor Technologies", "China"),
    ("ctrvision", "CTRvision", "China"),
    ("inspur", "Inspur", "China"),
    ("dalle molle institute for artificial intelligence research", "Dalle Molle Institute for Artificial Intelligence Research", "Switzerland"),
    ("cyberagent", "CyberAgent", "Japan"),
    ("yahoo", "Yahoo", "United States"),
    ("linkedin corporation", "LinkedIn", "United States"),
    ("tu wien", "Vienna University of Technology", "Austria"),
    ("vienna university of technology", "Vienna University of Technology", "Austria"),
    ("graz university of technology", "Graz University of Technology", "Austria"),
    ("curtin university of technology", "Curtin University", "Australia"),
    ("qilu university of technology", "Qilu University of Technology", "China"),
    ("shenzhen technology university", "Shenzhen Technology University", "China"),
    ("wenzhou university of technology", "Wenzhou University of Technology", "China"),
    ("fujian university of technology", "Fujian University of Technology", "China"),
    ("university of technology nuremberg", "Technische Universität Nürnberg", "Germany"),
    ("poznan university of technology", "Poznań University of Technology", "Poland"),
    ("google deepmind", "Google DeepMind", "United Kingdom"),
    ("deepmind", "Google DeepMind", "United Kingdom"),
    ("google research india", "Google Research India", "India"),
    ("google research israel", "Google", "Israel"),
    ("google research", "Google", "United States"),
    ("google brain", "Google", "United States"),
    ("google ai", "Google", "United States"),
    ("google", "Google", "United States"),
    ("alphabet", "Google", "United States"),
    ("waymo", "Waymo", "United States"),
    ("wing (alphabet)", "Google", "United States"),
    ("meta ai", "Meta AI", "United States"),
    ("meta platforms", "Meta AI", "United States"),
    ("meta reality labs", "Meta AI", "United States"),
    ("meta", "Meta AI", "United States"),
    ("facebook ai", "Meta AI", "United States"),
    ("facebook", "Meta AI", "United States"),
    ("instagram", "Meta AI", "United States"),
    ("whatsapp", "Meta AI", "United States"),
    ("amazon web services", "Amazon", "United States"),
    ("aws ai", "Amazon", "United States"),
    ("amazon", "Amazon", "United States"),
    ("microsoft research asia", "Microsoft Research Asia", "China"),
    ("microsoft research montreal", "Microsoft Research", "Canada"),
    ("microsoft research cambridge", "Microsoft Research", "United Kingdom"),
    ("microsoft research", "Microsoft", "United States"),
    ("microsoft", "Microsoft", "United States"),
    ("azure", "Microsoft", "United States"),
    ("openai", "OpenAI", "United States"),
    ("ibm research", "IBM Research", "United States"),
    ("ibm", "IBM Research", "United States"),
    ("apple", "Apple", "United States"),
    ("nvidia", "NVIDIA", "United States"),
    ("intel labs", "Intel Labs", "United States"),
    ("intel", "Intel", "United States"),
    ("salesforce research", "Salesforce Research", "United States"),
    ("salesforce", "Salesforce Research", "United States"),
    ("snap research", "Snap Research", "United States"),
    ("snap inc", "Snap Research", "United States"),
    ("adobe research", "Adobe Research", "United States"),
    ("adobe", "Adobe Research", "United States"),
    ("uber ai", "Uber", "United States"),
    ("uber", "Uber", "United States"),
    ("tesla", "Tesla", "United States"),
    ("qualcomm", "Qualcomm", "United States"),
    ("x (formerly twitter)", "X (formerly Twitter)", "United States"),
    ("twitter", "X (formerly Twitter)", "United States"),
    ("airbnb", "Airbnb", "United States"),
    ("netflix", "Netflix", "United States"),
    ("doordash", "DoorDash", "United States"),
    ("bytedance", "ByteDance", "China"),
    ("tiktok", "ByteDance", "China"),
    ("alibaba group", "Alibaba Group", "China"),
    ("alibaba", "Alibaba Group", "China"),
    ("alipay", "Ant Group", "China"),
    ("ant group", "Ant Group", "China"),
    ("baidu research", "Baidu Research", "China"),
    ("baidu", "Baidu Research", "China"),
    ("didi research", "DiDi", "China"),
    ("didi", "DiDi", "China"),
    ("jd ai", "JD.com", "China"),
    ("jd.com", "JD.com", "China"),
    ("jd ai research", "JD.com", "China"),
    ("meituan", "Meituan", "China"),
    ("sensetime", "SenseTime", "China"),
    ("huawei", "Huawei", "China"),
    ("tencent", "Tencent", "China"),
    ("oppo", "OPPO", "China"),
    ("vivo", "Vivo", "China"),
    ("xiaomi", "Xiaomi", "China"),
    ("lenovo", "Lenovo", "China"),
    ("360 ai", "360 AI Institute", "China"),
    ("ping an", "Ping An Technology", "China"),
    ("naver", "NAVER", "South Korea"),
    ("kakaobrain", "Kakao Brain", "South Korea"),
    ("kakao brain", "Kakao Brain", "South Korea"),
    ("samsung", "Samsung Research", "South Korea"),
    ("lg ai", "LG AI Research", "South Korea"),
    ("sony", "Sony AI", "Japan"),
    ("rakuten", "Rakuten Institute of Technology", "Japan"),
    ("preferred networks", "Preferred Networks", "Japan"),
    ("riken", "RIKEN", "Japan"),
    ("fujitsu", "Fujitsu", "Japan"),
    ("mercari", "Mercari", "Japan"),
    ("navinfo", "NavInfo", "China"),
    ("sift science", "Sift Science", "United States"),
    ("vector institute", "Vector Institute", "Canada"),
    ("mila", "Mila - Quebec AI Institute", "Canada"),
    ("amii", "Alberta Machine Intelligence Institute (Amii)", "Canada"),
    ("element ai", "ServiceNow Research", "Canada"),
    ("servicenow research", "ServiceNow Research", "Canada"),
    ("kunlun lab", "Kunlun Lab", "China"),
    ("noah s ark lab", "Huawei Noah's Ark Lab", "China"),
    ("huawei noah", "Huawei Noah's Ark Lab", "China"),
    ("allen institute for ai", "Allen Institute for AI", "United States"),
    ("allen ai", "Allen Institute for AI", "United States"),
    ("vector ai", "Vector Institute", "Canada"),
    ("isi", "USC Information Sciences Institute", "United States"),
    ("idiap", "Idiap Research Institute", "Switzerland"),
    ("max planck", "Max Planck Institute", "Germany"),
    ("helmholtz", "Helmholtz Association", "Germany"),
    ("fraunhofer", "Fraunhofer Society", "Germany"),
    ("tubingen ai center", "Tübingen AI Center", "Germany"),
    ("lmu munich", "Ludwig Maximilian University of Munich", "Germany"),
    ("tu munich", "Technical University of Munich", "Germany"),
    ("kaust", "King Abdullah University of Science and Technology", "Saudi Arabia"),
    ("kaist", "Korea Advanced Institute of Science & Technology", "South Korea"),
    ("postech", "Pohang University of Science & Technology", "South Korea"),
    ("skoltech", "Skolkovo Institute of Science and Technology", "Russia"),
    ("a star", "Agency for Science Technology and Research (A*STAR)", "Singapore"),
    ("a*star", "Agency for Science Technology and Research (A*STAR)", "Singapore"),
    ("singapore university of technology and design", "Singapore University of Technology and Design", "Singapore"),
    ("idiap research", "Idiap Research Institute", "Switzerland"),
    ("inria", "Inria", "France"),
    ("cea list", "CEA LIST", "France"),
    ("sorbonne", "Sorbonne Université", "France"),
    ("universite de montreal", "Université de Montréal", "Canada"),
    ("polytechnique montreal", "Polytechnique Montréal", "Canada"),
    ("mcgill", "McGill University", "Canada"),
    ("mcmaster university", "McMaster University", "Canada"),
    ("queens university", "Queen's University", "Canada"),
    ("oxford", "University of Oxford", "United Kingdom"),
    ("imperial college", "Imperial College London", "United Kingdom"),
    ("ucl", "University College London", "United Kingdom"),
    ("university college london", "University College London", "United Kingdom"),
    ("university of cambridge", "University of Cambridge", "United Kingdom"),
    ("edinburgh", "University of Edinburgh", "United Kingdom"),
    ("manchester", "University of Manchester", "United Kingdom"),
    ("birmingham", "University of Birmingham", "United Kingdom"),
    ("sheffield", "University of Sheffield", "United Kingdom"),
    ("bristol", "University of Bristol", "United Kingdom"),
    ("warwick", "University of Warwick", "United Kingdom"),
    ("swansea", "Swansea University", "United Kingdom"),
    ("university of leeds", "University of Leeds", "United Kingdom"),
    ("national university of defense technology", "National University of Defense Technology", "China"),
    ("beihang university", "Beihang University", "China"),
    ("beijing institute of technology", "Beijing Institute of Technology", "China"),
    ("nankai university", "Nankai University", "China"),
    ("sun yat sen university", "Sun Yat-sen University", "China"),
    ("xi an jiaotong university", "Xi'an Jiaotong University", "China"),
    ("xian jiaotong university", "Xi'an Jiaotong University", "China"),
    ("hunan university", "Hunan University", "China"),
    ("beijing normal university", "Beijing Normal University", "China"),
    ("beijing institute for general artificial intelligence", "BIGAI", "China"),
    ("cas", "Chinese Academy of Sciences", "China"),
    ("chinese academy of sciences", "Chinese Academy of Sciences", "China"),
    ("shenzhen institute of advanced technology", "Shenzhen Institute of Advanced Technology", "China"),
    ("the hong kong polytechnic university", "The Hong Kong Polytechnic University", "Hong Kong"),
    ("polytechnic university", "The Hong Kong Polytechnic University", "Hong Kong"),
    ("city university of hong kong", "City University of Hong Kong", "Hong Kong"),
    ("university of tokyo", "The University of Tokyo", "Japan"),
    ("kyoto university", "Kyoto University", "Japan"),
    ("osaka university", "Osaka University", "Japan"),
    ("tohoku university", "Tohoku University", "Japan"),
    ("tokyo institute of technology", "Tokyo Institute of Technology", "Japan"),
    ("keio university", "Keio University", "Japan"),
    ("nara institute of science and technology", "Nara Institute of Science and Technology", "Japan"),
    ("hanyang university", "Hanyang University", "South Korea"),
    ("yonsei university", "Yonsei University", "South Korea"),
    ("sogang university", "Sogang University", "South Korea"),
    ("university of melbourne", "University of Melbourne", "Australia"),
    ("university of sydney", "University of Sydney", "Australia"),
    ("monash university", "Monash University", "Australia"),
    ("unsw", "University of New South Wales", "Australia"),
    ("australian national university", "Australian National University", "Australia"),
    ("university of queensland", "University of Queensland", "Australia"),
    ("csiro", "CSIRO", "Australia"),
    ("data61", "CSIRO Data61", "Australia"),
    ("university of auckland", "University of Auckland", "New Zealand"),
    ("new york university", "New York University", "United States"),
    ("nyu", "New York University", "United States"),
    ("columbia university", "Columbia University", "United States"),
    ("princeton university", "Princeton University", "United States"),
    ("harvard university", "Harvard University", "United States"),
    ("harvard", "Harvard University", "United States"),
    ("yale university", "Yale University", "United States"),
    ("brown university", "Brown University", "United States"),
    ("university of chicago", "University of Chicago", "United States"),
    ("uchicago", "University of Chicago", "United States"),
    ("northwestern university", "Northwestern University", "United States"),
    ("duke university", "Duke University", "United States"),
    ("university of pennsylvania", "University of Pennsylvania", "United States"),
    ("upenn", "University of Pennsylvania", "United States"),
    ("johns hopkins university", "Johns Hopkins University", "United States"),
    ("jhu", "Johns Hopkins University", "United States"),
    ("university of wisconsin madison", "University of Wisconsin - Madison", "United States"),
    ("wisconsin madison", "University of Wisconsin - Madison", "United States"),
    ("university of maryland", "University of Maryland", "United States"),
    ("university of southern california", "University of Southern California", "United States"),
    ("usc", "University of Southern California", "United States"),
    ("university of north carolina", "University of North Carolina at Chapel Hill", "United States"),
    ("unc", "University of North Carolina at Chapel Hill", "United States"),
    ("cornell tech", "Cornell Tech", "United States"),
    ("cornell university", "Cornell University", "United States"),
    ("university of michigan", "University of Michigan - Ann Arbor", "United States"),
    ("university of colorado boulder", "University of Colorado Boulder", "United States"),
    ("rice university", "Rice University", "United States"),
    ("texas a and m university", "Texas A&M University", "United States"),
    ("texas a&m university", "Texas A&M University", "United States"),
    ("arizona state university", "Arizona State University", "United States"),
    ("university of arizona", "University of Arizona", "United States"),
    ("university of utah", "University of Utah", "United States"),
    ("virginia tech", "Virginia Tech", "United States"),
    ("university of virginia", "University of Virginia", "United States"),
    ("purdue university", "Purdue University", "United States"),
    ("ohio state university", "The Ohio State University", "United States"),
    ("north carolina state university", "North Carolina State University", "United States"),
    ("oregon state university", "Oregon State University", "United States"),
    ("university of alaska", "University of Alaska", "United States"),
    ("university of massachusetts amherst", "University of Massachusetts Amherst", "United States"),
    ("umass amherst", "University of Massachusetts Amherst", "United States"),
    ("university of rochester", "University of Rochester", "United States"),
    ("boston university", "Boston University", "United States"),
    ("tufts university", "Tufts University", "United States"),
    ("brandeis university", "Brandeis University", "United States"),
    ("georgetown university", "Georgetown University", "United States"),
    ("george washington university", "George Washington University", "United States"),
    ("george mason university", "George Mason University", "United States"),
    ("national taiwan university", "National Taiwan University", "Taiwan"),
    ("ntu taiwan", "National Taiwan University", "Taiwan"),
    ("academia sinica", "Academia Sinica", "Taiwan"),
    ("chunghua telecom laboratories", "Chunghwa Telecom Laboratories", "Taiwan"),
    ("university of twente", "University of Twente", "Netherlands"),
    ("delft university of technology", "Delft University of Technology", "Netherlands"),
    ("vu amsterdam", "Vrije Universiteit Amsterdam", "Netherlands"),
    ("radboud university", "Radboud University", "Netherlands"),
    ("wageningen university", "Wageningen University & Research", "Netherlands"),
    ("louvain", "Université catholique de Louvain", "Belgium"),
    ("ku leuven", "KU Leuven", "Belgium"),
    ("ghent university", "Ghent University", "Belgium"),
    ("politecnico di milano", "Politecnico di Milano", "Italy"),
    ("politecnico di torino", "Politecnico di Torino", "Italy"),
    ("sapienza university of rome", "Sapienza University of Rome", "Italy"),
    ("university of milano", "University of Milan", "Italy"),
    ("universitat pompeu fabra", "Universitat Pompeu Fabra", "Spain"),
    ("universidad autonoma de madrid", "Universidad Autónoma de Madrid", "Spain"),
    ("university of barcelona", "University of Barcelona", "Spain"),
    ("university of lisbon", "University of Lisbon", "Portugal"),
    ("tecnico lisboa", "Instituto Superior Técnico", "Portugal"),
    ("ensae", "ENSAE Paris", "France"),
    ("ens paris saclay", "ENS Paris-Saclay", "France"),
    ("paris dauphine", "Université Paris Dauphine", "France"),
    ("university of geneva", "University of Geneva", "Switzerland"),
    ("university of basel", "University of Basel", "Switzerland"),
    ("karlsruhe institute of technology", "Karlsruhe Institute of Technology", "Germany"),
    ("rwth aachen", "RWTH Aachen University", "Germany"),
    ("university of hamburg", "University of Hamburg", "Germany"),
    ("technical university of berlin", "Technical University of Berlin", "Germany"),
    ("charles university", "Charles University", "Czech Republic"),
    ("aalto university", "Aalto University", "Finland"),
    ("university of helsinki", "University of Helsinki", "Finland"),
    ("university of tampere", "Tampere University", "Finland"),
    ("university of oslo", "University of Oslo", "Norway"),
    ("ntnu", "Norwegian University of Science and Technology", "Norway"),
    ("kth royal institute of technology", "KTH Royal Institute of Technology", "Sweden"),
    ("chalmers university of technology", "Chalmers University of Technology", "Sweden"),
    ("lund university", "Lund University", "Sweden"),
    ("uppsala university", "Uppsala University", "Sweden"),
    ("universidade federal", "Federal University (Brazil)", "Brazil"),
    ("universidad nacional autonoma de mexico", "Universidad Nacional Autónoma de México", "Mexico"),
    ("puc rio", "Pontifical Catholic University of Rio de Janeiro", "Brazil"),
    ("fudan university", "Fudan University", "China"),
    ("sichuan university", "Sichuan University", "China"),
    ("tianjin university", "Tianjin University", "China"),
    ("huazhong university of science and technology", "Huazhong University of Science and Technology", "China"),
    ("south china university of technology", "South China University of Technology", "China"),
    ("xi dian university", "Xidian University", "China"),
    ("beijing jiaotong university", "Beijing Jiaotong University", "China"),
    ("hong kong baptist university", "Hong Kong Baptist University", "Hong Kong"),
    ("hong kong metropolitan university", "Hong Kong Metropolitan University", "Hong Kong"),
]

def _pattern_to_regex(pattern: str) -> Optional[Pattern[str]]:
    tokens = [re.escape(token) for token in pattern.split() if token]
    if not tokens:
        return None
    regex = r"\b" + r"\s+".join(tokens) + r"\b"
    return re.compile(regex)


MANUAL_SUBSTRING_RULES: List[Tuple[str, str, str, Pattern[str]]] = []
for pattern, canonical, country in RAW_MANUAL_SUBSTRINGS:
    normalized_pattern = _normalize_text(pattern)
    if not normalized_pattern:
        continue
    regex = _pattern_to_regex(normalized_pattern)
    if not regex:
        continue
    MANUAL_SUBSTRING_RULES.append((normalized_pattern, canonical, country, regex))

# Sort manual substring rules by pattern length (descending) to prefer specific matches.
MANUAL_SUBSTRING_RULES.sort(key=lambda rule: len(rule[0]), reverse=True)


ALL_COUNTRIES: Set[str] = {
    country for countries in REGION_COUNTRIES.values() for country in countries if country
}
ALL_COUNTRIES.update(COUNTRY_SYNONYMS.values())


@dataclass
class AffiliationMapping:
    canonical: str
    country: str
    region: str
    source: str
    confidence: float


class AffiliationResolver:
    """Resolves raw affiliation strings into canonical organization, country, and region."""

    def __init__(self, university_dataset_path: Path):
        self.university_df = pd.read_json(university_dataset_path)
        self.university_names: List[str] = self.university_df["name"].tolist()
        self.university_name_to_country: Dict[str, str] = dict(
            zip(self.university_df["name"], self.university_df["country"])
        )
        self.university_names_norm: Dict[str, str] = {
            _normalize_text(name): name for name in self.university_names
        }
        self.generic_tokens = {
            "inc",
            "incorporated",
            "co",
            "company",
            "corp",
            "corporation",
            "ltd",
            "llc",
            "limited",
            "ai",
            "ml",
            "lab",
            "labs",
            "research",
            "center",
            "centre",
            "dept",
            "department",
            "school",
        }

    def resolve(self, raw: str) -> AffiliationMapping:
        clean = raw.strip()
        norm = _normalize_text(clean)

        if not norm:
            return AffiliationMapping(
                canonical="Unknown", country="Unknown", region="Other/Unknown", source="empty", confidence=0.0
            )

        manual = self._apply_manual(norm, clean)
        if manual:
            return manual

        university = self._match_university(clean)
        if university:
            return university

        keyword = self._country_from_keywords(norm)
        if keyword:
            country = keyword
            region = self._country_to_region(country)
            canonical = _title_case(norm)
            return AffiliationMapping(
                canonical=canonical,
                country=country,
                region=region,
                source="keyword",
                confidence=0.5,
            )

        canonical = clean
        country = "Unknown"
        region = "Other/Unknown"
        return AffiliationMapping(
            canonical=canonical,
            country=country,
            region=region,
            source="unknown",
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Manual helpers
    # ------------------------------------------------------------------

    def _apply_manual(self, norm: str, clean: str) -> Optional[AffiliationMapping]:
        if norm in MANUAL_EXACT:
            canonical, country, source = MANUAL_EXACT[norm]
            region = self._country_to_region(country)
            return AffiliationMapping(
                canonical=canonical, country=country, region=region, source=source, confidence=1.0
            )

        for pattern, canonical, country, regex in MANUAL_SUBSTRING_RULES:
            if regex.search(norm):
                region = self._country_to_region(country)
                return AffiliationMapping(
                    canonical=canonical, country=country, region=region, source="manual_substring", confidence=0.9
                )
        return None

    def _match_university(self, clean: str) -> Optional[AffiliationMapping]:
        stripped = self._strip_corporate_suffix(clean)

        direct_match = process.extractOne(
            stripped,
            self.university_names,
            scorer=fuzz.WRatio,
            score_cutoff=90,
        )
        if direct_match:
            name, score, _ = direct_match
            country = self.university_name_to_country.get(name, "Unknown")
            region = self._country_to_region(country)
            return AffiliationMapping(
                canonical=name, country=country, region=region, source="university_dataset", confidence=score / 100.0
            )

        segments = re.split(r"[;/|]", stripped)
        segments = [segment.strip() for segment in segments if segment.strip()]
        segments = segments or [stripped]

        for segment in reversed(segments):
            pieces = [piece.strip() for piece in re.split(r",|\(|\)", segment) if piece.strip()]
            pieces = pieces or [segment]

            for piece in reversed(pieces):
                normalized_piece = _normalize_text(piece)
                if len(normalized_piece) < 4 or normalized_piece in self.generic_tokens:
                    continue
                match = process.extractOne(
                    piece,
                    self.university_names,
                    scorer=fuzz.WRatio,
                    score_cutoff=82,
                )
                if match:
                    name, score, _ = match
                    country = self.university_name_to_country.get(name, "Unknown")
                    region = self._country_to_region(country)
                    return AffiliationMapping(
                        canonical=name, country=country, region=region, source="university_dataset", confidence=score / 100.0
                    )
        return None

    def _country_from_keywords(self, norm: str) -> Optional[str]:
        for alias, country in COUNTRY_SYNONYMS.items():
            if alias in norm:
                return country

        for country in sorted(ALL_COUNTRIES, key=len, reverse=True):
            country_norm = _normalize_text(country)
            if not country_norm:
                continue
            if re.search(rf"\b{re.escape(country_norm)}\b", norm):
                return country
        return None

    def _country_to_region(self, country: str) -> str:
        for region, countries in REGION_COUNTRIES.items():
            if country in countries:
                return region
        return "Other/Unknown"

    def _strip_corporate_suffix(self, text: str) -> str:
        suffixes = [
            "inc.",
            "inc",
            "co., ltd.",
            "co., ltd",
            "co ltd",
            "co. ltd.",
            "co. ltd",
            "co.,ltd.",
            "co",
            "corp.",
            "corp",
            "corporation",
            "limited",
            "ltd.",
            "ltd",
            "llc",
            "gmbh",
            "s.r.l.",
            "srl",
            "s.l.",
            "s.a.",
            "pte. ltd.",
            "pte ltd",
            "ag",
            "bv",
            "oy",
            "ab",
        ]
        clean = text.strip()
        for suffix in suffixes:
            if clean.lower().endswith(f" {suffix}"):
                clean = clean[: -len(suffix) - 1]
        return clean


# --------------------------------------------------------------------------------------
# Data processing
# --------------------------------------------------------------------------------------


def load_neurips_affiliations() -> Tuple[Dict[str, Set[str]], Dict[int, List[Tuple[str, List[str]]]]]:
    files = sorted(PAPERTRAILS_DIR.glob("neurips20??_accepted.csv"))
    if not files:
        raise FileNotFoundError("No NeurIPS accepted CSV files found.")

    unique_affiliations: Dict[str, Set[str]] = defaultdict(set)
    per_year_rows: Dict[int, List[Tuple[str, List[str]]]] = defaultdict(list)

    for csv_path in files:
        match = re.search(r"(\d{4})", csv_path.stem)
        if not match:
            continue
        year = int(match.group(1))
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            paper_id = str(row.get("submission_number") or row.get("note_id") or row.get("id") or "")
            raw_affs: List[str] = []
            for aff in str(row.get("affiliations") or "").split(";"):
                aff = aff.strip()
                if aff:
                    raw_affs.append(aff)
                    unique_affiliations[aff].add(str(year))
            per_year_rows[year].append((paper_id, raw_affs))
    return unique_affiliations, per_year_rows


def build_affiliation_mappings() -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], pd.DataFrame, Dict[int, List[Tuple[str, List[str]]]]]:
    resolver = AffiliationResolver(DATA_DIR / "world_universities_and_domains.json")
    unique_affiliations, per_year_rows = load_neurips_affiliations()

    normalization: Dict[str, str] = {}
    canonical_metadata: Dict[str, Dict[str, str]] = {}
    records: List[Dict[str, object]] = []

    for raw in unique_affiliations:
        mapping = resolver.resolve(raw)
        normalization[raw] = mapping.canonical
        existing = canonical_metadata.get(mapping.canonical)
        if existing:
            # If we already have a record, prefer higher confidence or manual sources.
            if mapping.confidence > existing.get("confidence", 0):
                canonical_metadata[mapping.canonical] = {
                    "country": mapping.country,
                    "region": mapping.region,
                    "source": mapping.source,
                    "confidence": mapping.confidence,
                }
        else:
            canonical_metadata[mapping.canonical] = {
                "country": mapping.country,
                "region": mapping.region,
                "source": mapping.source,
                "confidence": mapping.confidence,
            }
        records.append(
            {
                "raw_affiliation": raw,
                "canonical_affiliation": mapping.canonical,
                "country": mapping.country,
                "region": mapping.region,
                "source": mapping.source,
                "confidence": mapping.confidence,
            }
        )

    df = pd.DataFrame(records)
    return normalization, canonical_metadata, df, per_year_rows


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------------------


def analyze(normalization: Dict[str, str], canonical_metadata: Dict[str, Dict[str, str]], per_year_rows):
    paper_records: List[Dict[str, object]] = []

    for year, rows in per_year_rows.items():
        for paper_id, raw_affs in rows:
            canonical_affs = []
            regions_all = set()
            regions_known = set()
            countries = set()

            for raw in raw_affs:
                canonical = normalization.get(raw)
                if not canonical:
                    continue
                canonical_affs.append(canonical)
                meta = canonical_metadata.get(canonical, {})
                region = meta.get("region", "Other/Unknown")
                country = meta.get("country", "Unknown")
                if region:
                    regions_all.add(region)
                    if region != "Other/Unknown":
                        regions_known.add(region)
                if country:
                    countries.add(country)

            canonical_affs = sorted(set(canonical_affs))
            countries = set(c for c in countries if c)

            paper_records.append(
                {
                    "year": year,
                    "paper_id": paper_id,
                    "affiliations": canonical_affs,
                    "regions_all": sorted(regions_all),
                    "regions_known": sorted(regions_known),
                    "countries": sorted(countries),
                }
            )

    papers_df = pd.DataFrame(paper_records)

    # Region counts per year (counting each paper once per region)
    region_records: List[Dict[str, object]] = []
    for _, row in papers_df.iterrows():
        year = row["year"]
        for region in row["regions_known"]:
            region_records.append({"year": year, "region": region})
    region_counts_df = pd.DataFrame(region_records)
    region_counts = region_counts_df.groupby(["year", "region"]).size().reset_index(name="paper_count")

    # Compute share per year
    totals = region_counts.groupby("year")["paper_count"].transform("sum")
    region_counts["share"] = region_counts["paper_count"] / totals

    # Top affiliations per year
    affiliation_records: List[Dict[str, object]] = []
    for _, row in papers_df.iterrows():
        year = row["year"]
        for aff in row["affiliations"]:
            affiliation_records.append({"year": year, "affiliation": aff})
    affiliation_df = pd.DataFrame(affiliation_records)
    aff_counts = affiliation_df.groupby(["year", "affiliation"]).size().reset_index(name="paper_count")
    aff_counts["region"] = aff_counts["affiliation"].map(lambda aff: canonical_metadata.get(aff, {}).get("region", "Other/Unknown"))
    aff_counts["country"] = aff_counts["affiliation"].map(lambda aff: canonical_metadata.get(aff, {}).get("country", "Unknown"))

    top_affiliations = (
        aff_counts.sort_values(["year", "paper_count"], ascending=[True, False])
        .groupby("year")
        .head(20)
        .reset_index(drop=True)
    )
    top_affiliations["rank"] = top_affiliations.groupby("year")["paper_count"].rank(
        method="min", ascending=False
    )

    # US vs China collaboration categories
    collaboration_records: List[Dict[str, object]] = []
    for _, row in papers_df.iterrows():
        year = row["year"]
        regions = set(row["regions_known"])
        has_us = "United States" in regions
        has_china = "China" in regions
        if has_us and has_china:
            category = "US-China collaboration"
        elif has_us:
            category = "US only"
        elif has_china:
            category = "China only"
        else:
            category = "Other regions"
        collaboration_records.append({"year": year, "category": category})
    collaboration_df = pd.DataFrame(collaboration_records)
    collaboration_counts = collaboration_df.groupby(["year", "category"]).size().reset_index(name="paper_count")
    total_by_year = collaboration_counts.groupby("year")["paper_count"].transform("sum")
    collaboration_counts["share"] = collaboration_counts["paper_count"] / total_by_year

    unknown_counts = (
        papers_df[papers_df["regions_known"].apply(len) == 0]
        .groupby("year")
        .size()
        .reset_index(name="paper_count")
    )

    return {
        "papers_df": papers_df,
        "region_counts": region_counts,
        "top_affiliations": top_affiliations,
        "collaboration_counts": collaboration_counts,
        "affiliation_counts": aff_counts,
        "unknown_counts": unknown_counts,
    }


# --------------------------------------------------------------------------------------
# Visualization helpers
# --------------------------------------------------------------------------------------


def plot_region_trends(region_counts: pd.DataFrame, path: Path) -> None:
    region_counts = region_counts.copy()
    region_counts = region_counts[region_counts["region"].isin(REGION_ORDER)]
    region_counts["region"] = pd.Categorical(region_counts["region"], categories=REGION_ORDER, ordered=True)
    region_counts = region_counts.dropna(subset=["region"])

    palette = sns.color_palette("Dark2", n_colors=len(REGION_ORDER))
    color_map = {region: palette[idx % len(palette)] for idx, region in enumerate(REGION_ORDER)}

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    for region in REGION_ORDER:
        subset = region_counts[region_counts["region"] == region]
        if subset.empty:
            continue
        plt.plot(
            subset["year"],
            subset["paper_count"],
            marker="o",
            linewidth=2.2,
            color=color_map[region],
            label=region,
        )
        last_row = subset.iloc[-1]
        plt.text(
            last_row["year"] + 0.05,
            last_row["paper_count"],
            f"{region} ({int(last_row['paper_count'])})",
            color=color_map[region],
            va="center",
            fontsize=9,
        )

    plt.xlim(region_counts["year"].min() - 0.2, region_counts["year"].max() + 1.0)
    plt.ylim(bottom=0)
    plt.title("NeurIPS Papers by Primary Region (Unique Papers, 2021-2025)")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_region_share(region_counts: pd.DataFrame, path: Path) -> None:
    share_df = region_counts.copy()
    share_df = share_df[share_df["region"].isin(REGION_ORDER)]
    share_df["region"] = pd.Categorical(share_df["region"], categories=REGION_ORDER, ordered=True)
    share_df = share_df.dropna(subset=["region"])

    palette = sns.color_palette("Dark2", n_colors=len(REGION_ORDER))
    color_map = {region: palette[idx % len(palette)] for idx, region in enumerate(REGION_ORDER)}

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = plt.gca()

    for region in REGION_ORDER:
        subset = share_df[share_df["region"] == region]
        if subset.empty:
            continue
        ax.plot(
            subset["year"],
            subset["share"],
            marker="o",
            linewidth=2.2,
            color=color_map[region],
            label=region,
        )
        last_row = subset.iloc[-1]
        ax.text(
            last_row["year"] + 0.05,
            last_row["share"],
            f"{region} ({last_row['share']:.0%})",
            color=color_map[region],
            va="center",
            fontsize=9,
        )

    ax.set_xlim(share_df["year"].min() - 0.2, share_df["year"].max() + 1.0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Share of NeurIPS Papers by Region (2021-2025)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of Papers")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_region_share_stacked(region_counts: pd.DataFrame, path: Path) -> None:
    share_df = region_counts.copy()
    share_df = share_df[share_df["region"].isin(REGION_ORDER)]
    share_df["region"] = pd.Categorical(share_df["region"], categories=REGION_ORDER, ordered=True)
    share_df = share_df.dropna(subset=["region"])

    pivot = share_df.pivot_table(
        index="year",
        columns="region",
        values="share",
        fill_value=0.0,
        observed=False,
    ).sort_index()

    years = pivot.index.values
    present_regions = [region for region in REGION_ORDER if region in pivot.columns]

    palette = sns.color_palette("Dark2", n_colors=len(present_regions))
    color_map = {region: palette[idx % len(palette)] for idx, region in enumerate(present_regions)}

    plt.figure(figsize=(11, 6))
    bottom = np.zeros_like(years, dtype=float)
    for region in present_regions:
        values = pivot[region].values
        plt.bar(
            years,
            values,
            bottom=bottom,
            color=color_map[region],
            label=region,
        )

        label_positions = bottom + values / 2
        for x, y, val in zip(years, label_positions, values):
            if val <= 0 or region == "Latin America":
                continue
            label = f"{region}"
            if region in {"Canada", "Middle East & Africa"}:
                label += f" {val:.0%}"
            elif val >= 0.08:
                label += f"\n{val:.0%}"
            text_color = "white" if val >= 0.08 else "black"
            plt.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

        bottom = bottom + values

    if "Latin America" in present_regions:
        latin_top = pivot[present_regions].cumsum(axis=1)["Latin America"].values
        latin_share = pivot["Latin America"].values
        for x, y, share in zip(years, latin_top, latin_share):
            if share <= 0:
                continue
            plt.text(
                x,
                y + 0.015,
                f"Latin America {share:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )

    ax = plt.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title("Regional Share of NeurIPS Papers", fontsize=16, fontweight="bold", pad=18)
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share of Papers", fontsize=12, fontweight="bold")
    ax.set_facecolor("#f9f9fb")
    ax.grid(color="#e5e5ef", linestyle="--", linewidth=0.7, alpha=0.6, axis="y")
    ax.tick_params(axis="both", labelsize=10, colors="#444444")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def select_affiliations_for_bump(
    aff_counts: pd.DataFrame, canonical_metadata: Dict[str, Dict[str, str]]
) -> Tuple[pd.DataFrame, List[str]]:
    years = sorted(aff_counts["year"].unique())
    filtered = aff_counts[aff_counts["region"].isin(REGION_ORDER)].copy()
    pivot = filtered.pivot_table(index="affiliation", columns="year", values="paper_count", fill_value=0)
    region_map = {aff: canonical_metadata.get(aff, {}).get("region", "Other/Unknown") for aff in pivot.index}

    selected: List[str] = []
    for year in years:
        yearly = (
            filtered[filtered["year"] == year]
            .sort_values("paper_count", ascending=False)
            .head(7)["affiliation"]
            .tolist()
        )
        for aff in yearly:
            if aff not in selected:
                selected.append(aff)

    # Ensure key 2025 Chinese affiliations are present
    china_2025 = (
        filtered[(filtered["year"] == years[-1]) & (filtered["region"] == "China")]
        .sort_values("paper_count", ascending=False)
    )
    for aff in china_2025["affiliation"].head(6):
        if aff not in selected:
            selected.append(aff)

    # Ensure the list remains manageable by keeping top overall contributors if needed
    if len(selected) > 20:
        totals = pivot.loc[selected].sum(axis=1).sort_values(ascending=False)
        selected = [aff for aff in totals.index[:20]]

    records: List[Dict[str, object]] = []
    for aff in selected:
        region = region_map.get(aff, "Other/Unknown")
        for year in years:
            count = pivot.loc[aff, year] if year in pivot.columns else 0
            records.append(
                {
                    "affiliation": aff,
                    "year": year,
                    "paper_count": count,
                    "region": region,
                }
            )
    plot_df = pd.DataFrame(records)
    return plot_df, selected


def plot_top_affiliations(
    aff_counts: pd.DataFrame, canonical_metadata: Dict[str, Dict[str, str]], path: Path
) -> None:
    plot_df, selected_affiliations = select_affiliations_for_bump(aff_counts, canonical_metadata)
    years = sorted(plot_df["year"].unique())

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    ax = plt.gca()

    present_regions = sorted(plot_df["region"].unique(), key=lambda r: REGION_ORDER.index(r))
    palette = sns.color_palette("tab10", n_colors=len(present_regions))
    color_map = {region: palette[idx % len(palette)] for idx, region in enumerate(present_regions)}

    left_positions: List[float] = []
    right_positions: List[float] = []

    for affiliation in selected_affiliations:
        subset = plot_df[plot_df["affiliation"] == affiliation].sort_values("year")
        region = subset["region"].iloc[0]
        color = color_map.get(region, "gray")

        ax.plot(
            subset["year"],
            subset["paper_count"],
            marker="o",
            linewidth=2,
            color=color,
        )

        start_val = subset.iloc[0]["paper_count"]
        end_val = subset.iloc[-1]["paper_count"]

        left_y, left_arrow = allocate_label_position(left_positions, start_val)
        right_y, right_arrow = allocate_label_position(right_positions, end_val)

        left_kwargs = dict(
            xy=(years[0], start_val),
            xytext=(years[0] - (0.45 if left_arrow else 0.25), left_y),
            textcoords="data",
            ha="right",
            va="center",
            fontsize=9,
            color=color,
        )
        if left_arrow:
            left_kwargs["arrowprops"] = dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7)
        ax.annotate(f"{affiliation} ({int(start_val)})", **left_kwargs)

        right_kwargs = dict(
            xy=(years[-1], end_val),
            xytext=(years[-1] + (0.45 if right_arrow else 0.25), right_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=9,
            color=color,
        )
        if right_arrow:
            right_kwargs["arrowprops"] = dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7)
        ax.annotate(f"{affiliation} ({int(end_val)})", **right_kwargs)

    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.set_ylim(bottom=0)
    ax.set_xticks(years)
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("Dominant NeurIPS Affiliations, 2021-2025", fontsize=16, fontweight="bold", pad=16)

    handles = [Line2D([0], [0], color=color_map[region], lw=3) for region in present_regions]
    legend = ax.legend(
        handles,
        present_regions,
        title="Region",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.85),
        frameon=False,
    )

    ax.set_facecolor("#f9f9fb")
    ax.grid(color="#e5e5ef", linestyle="--", linewidth=0.7, alpha=0.7)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, colors="#444444")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_us_china_collaboration(collab_df: pd.DataFrame, path: Path) -> None:
    categories = ["US only", "China only", "US-China collaboration", "Other regions"]
    collab_df = collab_df.copy()
    collab_df["category"] = pd.Categorical(collab_df["category"], categories=categories, ordered=True)
    collab_df = collab_df.sort_values(["year", "category"])

    pivot = collab_df.pivot_table(
        index="year", columns="category", values="share", fill_value=0.0, observed=False
    )
    colors = sns.color_palette("Set2", n_colors=len(categories))
    color_map = {category: colors[idx] for idx, category in enumerate(categories)}

    plt.figure(figsize=(10, 6))
    bottom = None
    for category in categories:
        values = pivot[category]
        plt.bar(
            values.index,
            values.values,
            bottom=bottom,
            color=color_map[category],
            label=category,
        )
        bottom = values if bottom is None else bottom + values

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(0, 1)
    plt.title("Share of NeurIPS Papers by US/China Collaboration Category")
    plt.xlabel("Year")
    plt.ylabel("Share of Papers")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------


def main() -> None:
    normalization, canonical_metadata, mapping_df, per_year_rows = build_affiliation_mappings()

    save_json(normalization, BASE_DIR / "affiliation_normalization.json")
    save_json(canonical_metadata, BASE_DIR / "affiliation_region_mapping.json")
    save_csv(mapping_df, BASE_DIR / "affiliation_mapping_long.csv")

    analysis_outputs = analyze(normalization, canonical_metadata, per_year_rows)

    save_csv(analysis_outputs["region_counts"], BASE_DIR / "region_paper_counts.csv")
    save_csv(analysis_outputs["top_affiliations"], BASE_DIR / "top_affiliations_by_year.csv")
    save_csv(analysis_outputs["collaboration_counts"], BASE_DIR / "us_china_collaboration.csv")
    save_csv(analysis_outputs["affiliation_counts"], BASE_DIR / "affiliation_counts_full.csv")
    save_csv(analysis_outputs["unknown_counts"], BASE_DIR / "unknown_region_papers.csv")

    plot_region_trends(analysis_outputs["region_counts"], FIGURES_DIR / "region_trends.png")
    plot_region_share(analysis_outputs["region_counts"], FIGURES_DIR / "region_share.png")
    plot_region_share_stacked(analysis_outputs["region_counts"], FIGURES_DIR / "region_share_stacked.png")
    plot_top_affiliations(
        analysis_outputs["affiliation_counts"],
        canonical_metadata,
        FIGURES_DIR / "top_affiliation_bump.png",
    )
    plot_us_china_collaboration(analysis_outputs["collaboration_counts"], FIGURES_DIR / "us_china_collaboration.png")

    # Console summary to help validate coverage.
    coverage = mapping_df["region"].value_counts(dropna=False, normalize=True)
    print("Affiliation region coverage:")
    print((coverage * 100).round(2).to_string())
    unknown_total = int(analysis_outputs["unknown_counts"]["paper_count"].sum())
    print(f"Papers with no region assignment after normalization: {unknown_total}")


if __name__ == "__main__":
    main()
