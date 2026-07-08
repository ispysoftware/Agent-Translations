"""
Pre-publish validator for translation files (app + web schemas).
Usage:
  python validate.py            # validates the folder the script lives in
  python validate.py ..\\web    # validates another folder of <code>.json files
Exit code 1 on hard failures.
"""
import json
import re
import sys
import glob
import os

TOKEN_RE = re.compile(r"\[[A-Z0-9_]+\]|\{[^{}]+\}")
# Restricted to legal URL characters so CJK/Hangul text glued to a URL isn't swallowed
URL_RE = re.compile(r"https?://[A-Za-z0-9\-._~:/?#@!$&*+;=%()\[\]]+")
URL_STRIP = '.,;:)]}/'
BRAND_RE = re.compile(r"Agent ?DVR")

LATIN_LANGS = {'cs', 'da', 'de', 'es', 'fi', 'fr', 'hu', 'id', 'it', 'nb', 'nl', 'pl', 'pt', 'sv', 'tr', 'vi'}
FOREIGN_SCRIPT = re.compile(r'[Ѐ-ӿ؀-ۿऀ-ॿঀ-৿぀-ヿ一-鿿가-힯]')

EXPECTED_NAMES = {
    'ar': 'عربي', 'bn': 'বাংলা', 'cs': 'Čeština', 'da': 'Dansk', 'de': 'Deutsch',
    'es': 'Español', 'fa': 'فارسی', 'fi': 'Suomi', 'fr': 'Français', 'hi': 'हिंदी',
    'hu': 'Magyar', 'id': 'Bahasa Indonesia', 'it': 'Italiano', 'ja': '日本',
    'ko': '한국어', 'nl': 'Nederlands', 'nb': 'Norsk', 'pl': 'Polski',
    'pt': 'Português', 'ru': 'Русский', 'sv': 'Svenska', 'tr': 'Türkçe',
    'uk': 'українська', 'vi': 'Tiếng Việt', 'zh-cn': '中文', 'zh-tw': '繁体中文',
}


def norm_urls(text):
    return sorted(u.rstrip(URL_STRIP) for u in URL_RE.findall(text))


def flatten(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + (k,)))
        elif isinstance(v, str):
            out[prefix + (k,)] = v
    return out


def load(path):
    """Returns (flat_strings, header_dict_or_None). Handles both schemas."""
    with open(path, encoding='utf-8') as f:
        raw = f.read()
    if '\x00' in raw or '�' in raw:
        raise ValueError("null bytes or replacement chars in file")
    data = json.loads(raw)
    if isinstance(data, dict) and 'translations' in data:
        hdr = {k: v for k, v in data.items() if k != 'translations'}
        return flatten(data['translations']), hdr
    return flatten(data), None


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    en_flat, _ = load(os.path.join(folder, 'en.json'))

    hard_errors = 0
    print(f"{'lang':6} {'keys':>5} {'tokens':>7} {'urls':>5} {'brand':>6} {'script':>7} {'english':>8}")
    for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
        code = os.path.basename(path)[:-5]
        if code == 'en':
            continue
        try:
            t, hdr = load(path)
        except Exception as e:
            print(f"{code:6} FATAL: {e}")
            hard_errors += 1
            continue

        if hdr is not None:
            if hdr.get('culturecode') != code:
                print(f"{code:6} FATAL: culturecode is {hdr.get('culturecode')!r}")
                hard_errors += 1
            if EXPECTED_NAMES.get(code) and hdr.get('name') != EXPECTED_NAMES[code]:
                print(f"{code:6} WARN: name is {hdr.get('name')!r}, expected {EXPECTED_NAMES[code]!r}")

        missing = [k for k in en_flat if k not in t]
        extra = [k for k in t if k not in en_flat]
        if missing or extra:
            print(f"{code:6} FATAL: {len(missing)} missing keys {['/'.join(k) for k in missing[:3]]}, "
                  f"{len(extra)} extra keys {['/'.join(k) for k in extra[:3]]}")
            hard_errors += 1

        tok = url = brand = script = english = 0
        for k, v in t.items():
            e = en_flat.get(k)
            if e is None:
                continue
            if v == e:
                if any(c.isalpha() for c in TOKEN_RE.sub('', e)):
                    english += 1
                continue
            orig = TOKEN_RE.findall(e)
            oset = set(orig)
            if sorted(orig) != sorted(x for x in TOKEN_RE.findall(v) if x in oset):
                print(f"{code:6} TOKEN [{'/'.join(k)}]: {v[:70]!r}")
                tok += 1
            if norm_urls(e) != norm_urls(v):
                print(f"{code:6} URL   [{'/'.join(k)}]: {v[:70]!r}")
                url += 1
            if len(BRAND_RE.findall(v)) < len(BRAND_RE.findall(e)):
                print(f"{code:6} BRAND [{'/'.join(k)}]: {v[:70]!r}")
                brand += 1
            if code in LATIN_LANGS and FOREIGN_SCRIPT.search(v):
                print(f"{code:6} SCRIPT[{'/'.join(k)}]: {v[:70]!r}")
                script += 1

        hard_errors += tok + url + brand + script
        print(f"{code:6} {len(t):>5} {tok:>7} {url:>5} {brand:>6} {script:>7} {english:>8}")

    print()
    if hard_errors:
        print(f"FAILED: {hard_errors} hard error(s) - do not publish")
        sys.exit(1)
    print("ALL CHECKS PASSED - safe to publish")
    print("(the 'english' column is informational: strings identical to the source)")


if __name__ == "__main__":
    main()
