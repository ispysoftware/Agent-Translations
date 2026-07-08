import argparse
import json
import os
import re
import sys
import ollama

# --- DEFAULTS ---
DEFAULT_INPUT = "en.json"
DEFAULT_CHUNK_SIZE = 40
OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 8192}
MAX_RETRIES = 2          # retries per chunk before splitting
MIN_SPLIT_SIZE = 5       # don't split chunks smaller than this

# --- MODEL SELECTION ---
# Verify exact tags at https://ollama.com/library before first run.
DEFAULT_MODEL = "gemma4:26b"          # broad multilingual coverage, MoE = fast on 16GB VRAM
MODEL_OVERRIDES = {                    # per-language exceptions (Qwen leads on CJK)
    "ja": "qwen3.5:27b",
    "ko": "qwen3.5:27b",
    "zh-cn": "qwen3.5:27b",
    "zh-tw": "qwen3.5:27b",
}


def pick_model(lang_code, cli_model):
    """CLI -m overrides everything; otherwise per-language table, then default."""
    return cli_model or MODEL_OVERRIDES.get(lang_code, DEFAULT_MODEL)


# Top-level keys that must NEVER be sent to the LLM.
# They are set explicitly from the LANGUAGES table below.
IMMUTABLE_KEYS = {"culturecode", "name"}

# Keys whose reviewed value INTENTIONALLY equals the English (feature names,
# terms kept in English on purpose). Normally an English-identical value is
# treated as "untranslated" and retried every run — which re-rolls these into
# unwanted translations (e.g. "Timelapse" -> "Accéléré"). Pinning trusts the
# existing value in update mode. Initial generation is unaffected: if the key
# is missing from the output file it is still translated normally.
# Nested keys use "/" separators, e.g. "tt/recap".
PINNED_KEYS = {"timelapse", "appconnect"}

# Language code -> (English name for the prompt, native name for the "name" field).
# Matches the website's language list.
LANGUAGES = {
    "ar": ("Modern Standard Arabic", "عربي"),
    "bn": ("Bengali", "বাংলা"),
    "cs": ("Czech", "Čeština"),
    "da": ("Danish", "Dansk"),
    "de": ("German (Germany)", "Deutsch"),
    "es": ("European Spanish (Spain)", "Español"),
    "fa": ("Farsi (Persian)", "فارسی"),
    "fi": ("Finnish", "Suomi"),
    "fr": ("French (France)", "Français"),
    "hi": ("Hindi", "हिंदी"),
    "hu": ("Hungarian", "Magyar"),
    "id": ("Indonesian", "Bahasa Indonesia"),
    "it": ("Italian", "Italiano"),
    "ja": ("Japanese", "日本"),
    "ko": ("Korean", "한국어"),
    "nl": ("Dutch (Netherlands)", "Nederlands"),
    "nb": ("Norwegian (Bokmål)", "Norsk"),
    "pl": ("Polish", "Polski"),
    "pt": ("Brazilian Portuguese", "Português"),
    "ru": ("Russian", "Русский"),
    "sv": ("Swedish", "Svenska"),
    "tr": ("Turkish", "Türkçe"),
    "uk": ("Ukrainian", "українська"),
    "vi": ("Vietnamese", "Tiếng Việt"),
    "zh-cn": ("Simplified Chinese (Mandarin)", "中文"),
    "zh-tw": ("Traditional Chinese (Taiwanese Mandarin)", "繁体中文"),
}
ALL_LANGUAGES = list(LANGUAGES)

TOKEN_RE = re.compile(r"\[[A-Z0-9_]+\]|\{[^{}]+\}")  # [IPADDRESS] + {0}, {0:yyyy-MM-dd}, {DIR}, {C} styles
URL_RE = re.compile(r"https?://[A-Za-z0-9\-._~:/?#@!$&*+;=%()\[\]]+")


def ensure_model_installed(model_name):
    """Verifies the model is local before beginning."""
    print(f"🤖 Checking local model state for '{model_name}'...")
    try:
        local_models = ollama.list()
        models = [m['model'] for m in local_models.get('models', [])]
        if model_name in models or f"{model_name}:latest" in models:
            print(f" -> '{model_name}' is ready.")
            return
        print(f" -> '{model_name}' not found. Downloading via Ollama...")
        for progress in ollama.pull(model_name, stream=True):
            total = getattr(progress, 'total', None) or 0
            completed = getattr(progress, 'completed', None) or 0
            status = getattr(progress, 'status', '')
            if total:
                pct = completed / total * 100
                bar = "█" * int(pct // 4) + "░" * (25 - int(pct // 4))
                print(f"\r    [{bar}] {pct:5.1f}%  ({completed / 1e9:.2f}/{total / 1e9:.2f} GB) {status:<20}", end="", flush=True)
            else:
                print(f"\r    {status:<70}", end="", flush=True)
        print("\n -> Download complete!")
    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}\nIs the desktop app open?")
        sys.exit(1)


def collect_translatable(data):
    """Collect string refs, skipping immutable top-level keys (culturecode, name)."""
    refs = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k in IMMUTABLE_KEYS:
                continue
            if isinstance(v, str):
                refs.append((data, k, v))
            else:
                collect_strings(v, refs)
    else:
        collect_strings(data, refs)
    return refs


def collect_paths(data, prefix=(), out=None, top=True):
    """Map of path-tuple -> string value for every translatable string (immutable top-level keys skipped)."""
    if out is None:
        out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if top and k in IMMUTABLE_KEYS:
                continue
            if isinstance(v, str):
                out[prefix + (k,)] = v
            else:
                collect_paths(v, prefix + (k,), out, False)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, str):
                out[prefix + (i,)] = v
            else:
                collect_paths(v, prefix + (i,), out, False)
    return out


def collect_strings(data, string_list):
    """Recursively finds all strings and stores references to their exact memory locations."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                string_list.append((data, k, v))
            else:
                collect_strings(v, string_list)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, str):
                string_list.append((data, idx, item))
            else:
                collect_strings(item, string_list)


def needs_translation(text):
    """Skip strings with no translatable content (empty, numbers, pure tokens/symbols)."""
    stripped = TOKEN_RE.sub("", text).strip()
    return any(c.isalpha() for c in stripped)


def _norm_urls(text):
    """Extract URLs, stripping trailing punctuation that isn't part of the URL."""
    return sorted(u.rstrip('.,;:)]}/') for u in URL_RE.findall(text))


def urls_preserved(original, translated):
    """
    Every URL in the source must appear byte-identical in the translation,
    and the translation must not invent URLs. Models frequently corrupt
    domains, ports and IP addresses mid-URL when translating.
    """
    return _norm_urls(original) == _norm_urls(translated)


def tokens_preserved(original, translated):
    """
    Check that every [TOKEN] / {0} placeholder from the source survived translation
    intact (same tokens, same counts). Tokens the translation introduces on its own
    are allowed — e.g. Japanese writes button names as [OK] by convention.
    """
    orig = TOKEN_RE.findall(original)
    orig_set = set(orig)
    trans = [t for t in TOKEN_RE.findall(translated) if t in orig_set]
    return sorted(orig) == sorted(trans)


def build_system_prompt(target_lang):
    return f"""
    You are a precise technical translator for Video Surveillance (VMS) and Security Software.
    Translate the VALUES of this JSON object into {target_lang}.

    DOMAIN CONTEXT & BRAND PROTECTION:
    1. The software is a technical Video Surveillance platform.
    2. CRITICAL BRAND RULE: "Agent DVR" is a proprietary brand name. NEVER translate, transliterate, or alter "Agent DVR". It must remain exactly "Agent DVR" in all languages.
    3. "Cloud" refers to cloud-based servers/infrastructure, NOT weather.
    4. "Cookie" refers to web browser cookies. Do NOT translate it as a food or biscuit.
    5. "Stream" or "Feed" refers to live video surveillance camera data feeds.
    6. Maintain a professional, clean, B2B software engineering tone.

    UI & STYLE RULES:
    1. These are UI strings (labels, buttons, tooltips, error messages) from a desktop/web application. Keep translations concise — similar length to the English wherever possible, since they must fit in the same UI space.
    2. Use the FORMAL register consistently (e.g. "vous" in French, "Sie" in German, "usted" in Spanish, formal Korean/Japanese).
    3. TERMINOLOGY CONSISTENCY: translate the same English term identically every time (e.g. "camera", "recording", "alert", "device" must each map to one fixed translation).
    4. Never translate technical acronyms, protocols, formats, or third-party product names: RTSP, ONVIF, MQTT, HTTP, URL, IP, FPS, H.264, H.265, MP4, MJPEG, PTZ, Telegram, Dropbox, Google Drive, YouTube, etc.
    5. Preserve punctuation and separators exactly: trailing colons ":", ellipses "...", and "- " used as a separator must remain in place.
    6. CAPITALIZATION — THIS IS FREQUENTLY DONE WRONG, PAY ATTENTION: follow the TARGET language's own conventions, not English's. Nearly all European languages (French, Spanish, Portuguese, Italian, Czech, Danish, Finnish, Swedish, Norwegian, Dutch, Polish, Hungarian, Turkish...) use sentence case for UI labels: only the first word is capitalized. "General Settings" -> "Paramètres généraux" NOT "Paramètres Généraux"; "Save Thumbnails" -> "Salvar miniaturas" NOT "Salvar Miniaturas"; "Sound Detected" -> "Som detectado" NOT "Som Detectado". EXCEPTION: German capitalizes all nouns as its grammar requires. Only capitalize other mid-string words if they are proper nouns, references to named UI sections, or protected acronyms.
    7. STATE WORDS: translate UI state words like ON, OFF, Enabled, Disabled, Yes, No into the target language (e.g. French "Activé"/"Désactivé"). Do not leave them in English, even when uppercase.
    8. LOCALIZE UNITS & COMMON IT TERMS per target-language convention: e.g. in French "KB"/"kb" -> "ko", "MB" -> "Mo", "GB" -> "Go", "logs" -> "journaux". Protocol names and acronyms from rule 4 still stay unchanged.

    INPUT FORMAT:
    You receive a JSON object like {{"0": {{"key": "json.servername", "text": "Name"}}, "1": ...}}.
    - "text" is the English string to translate.
    - "key" is the app's internal identifier for that string. Use it ONLY as context to resolve ambiguity (e.g. key "json.servername" tells you "Name" is a label for a server's name, so translate it as a noun). NEVER translate, alter, or output the key.

    STRICT OPERATIONAL DIRECTIVES:
    1. Keep the numeric IDs ("0", "1", "2", etc.) EXACTLY identical. Do not alter them.
    2. VARIABLE & TOKEN PROTECTION: Text may contain tokens in square brackets like [LANG], [IPADDRESS], [PORT], [USERNAME] or numbered placeholders like {{0}}. Treat these as code literals. Do NOT translate or modify them. Leave them exactly as they are.
    3. Output ONLY a flat JSON object mapping each numeric ID directly to its translated string, like {{"0": "<translation>", "1": "<translation>"}}. Do NOT output nested objects, do NOT include "key" or "text" fields, no markdown backticks (```json), no conversational filler.
    """


def call_llm(model_name, system_prompt, payload, options=None):
    """Single LLM call. Returns dict of id -> translation. Raises on bad output."""
    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        format="json",
        options=options or OLLAMA_OPTIONS,
    )
    try:
        # Disable reasoning on thinking models (qwen3.5 etc.) — much faster,
        # and translation doesn't benefit from chain-of-thought.
        response = ollama.chat(**kwargs, think=False)
    except Exception:
        # Model or client doesn't support the think parameter
        response = ollama.chat(**kwargs)
    content_str = response.message.content if hasattr(response, 'message') else response['message']['content']
    if not content_str:
        raise ValueError("Empty response from model")
    try:
        return json.loads(content_str)
    except json.JSONDecodeError as e:
        context = content_str[max(0, e.pos - 60):e.pos + 30]
        raise ValueError(f"Invalid JSON ({e.msg} at char {e.pos}): ...{context!r}...")


def translate_texts(texts, target_lang, model_name, cache, failures, key_hints=None, depth=0):
    """
    Translate a list of unique strings, filling `cache` (original -> translated).
    Validates each result; retries failures, then splits the batch in half.
    Strings that could not be translated are added to `failures`.
    `key_hints` maps original text -> its JSON key, sent as disambiguation context.
    """
    key_hints = key_hints or {}
    pending = [t for t in texts if t not in cache]
    if not pending:
        return

    system_prompt = build_system_prompt(target_lang)

    for attempt in range(MAX_RETRIES):
        payload = {str(i): {"key": key_hints.get(t, ""), "text": t} for i, t in enumerate(pending)}
        opts = dict(OLLAMA_OPTIONS)
        if attempt > 0:
            # temperature 0 is deterministic: an identical retry would fail identically.
            # Add a little sampling noise to escape the failure mode.
            opts.update(temperature=0.4, seed=42 + attempt)
        try:
            translated_map = call_llm(model_name, system_prompt, payload, opts)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   ⚠️ Bad model output (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            continue
        except Exception as e:
            print(f"   ⚠️ Ollama error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            continue

        failed = []
        for i, original in enumerate(pending):
            value = translated_map.get(str(i))
            if value is None:
                # Model sometimes keys output by the hint key instead of the numeric ID
                hint = key_hints.get(original)
                if hint:
                    value = translated_map.get(hint)
            if isinstance(value, dict):
                # Model sometimes echoes the input object shape {"key":..., "text":...}
                value = value.get("text")
            if not isinstance(value, str) or not value.strip():
                failed.append(original)
            elif not tokens_preserved(original, value):
                print(f"   ⚠️ Token mismatch, will retry: {original[:60]!r}")
                failed.append(original)
            elif not urls_preserved(original, value):
                print(f"   ⚠️ URL corrupted, will retry: {original[:60]!r}")
                failed.append(original)
            else:
                cache[original] = value

        if not failed:
            return
        pending = failed  # retry only the failures

    # Retries exhausted: split and recurse, or give up on tiny batches
    if len(pending) > MIN_SPLIT_SIZE:
        mid = len(pending) // 2
        print(f"   ✂️ Splitting {len(pending)} stubborn strings into two smaller batches...")
        translate_texts(pending[:mid], target_lang, model_name, cache, failures, key_hints, depth + 1)
        translate_texts(pending[mid:], target_lang, model_name, cache, failures, key_hints, depth + 1)
    else:
        for original in pending:
            print(f"   ❌ Giving up, keeping English: {original[:60]!r}")
            cache[original] = original
            failures.add(original)


def load_existing_translations(output_file, src_paths):
    """
    Update mode: reuse translations from an existing output file, matched by JSON path.
    Returns (cache, new_count, removed_count).
    A previous value is trusted as a translation only if it differs from the current
    English text — identical values (untranslated/failed earlier) get retranslated.
    New keys are translated; keys no longer in the source drop out automatically
    because the output is rebuilt from the source structure.
    """
    if not os.path.exists(output_file):
        return {}, len(src_paths), 0
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            prev_paths = collect_paths(json.load(f))
    except Exception as e:
        print(f" -> Could not read existing '{output_file}' ({e}); regenerating from scratch.")
        return {}, len(src_paths), 0

    cache = {}
    new_count = 0
    for path, orig in src_paths.items():
        prev = prev_paths.get(path)
        pinned = "/".join(str(p) for p in path) in PINNED_KEYS
        if isinstance(prev, str) and (prev != orig or pinned):
            cache[orig] = prev
        elif prev is None:
            new_count += 1
    removed_count = sum(1 for p in prev_paths if p not in src_paths)
    return cache, new_count, removed_count


def translate_language(lang_code, args):
    english_name, native_name = LANGUAGES.get(lang_code, (lang_code, None))
    model = pick_model(lang_code, args.model)
    output_file = args.output or f"{lang_code}.json"

    print(f"\n🌍 Translating to {english_name} -> '{output_file}' (model: {model})")
    print(f"📖 Loading '{args.input}'...")
    with open(args.input, 'r', encoding='utf-8') as f:
        source_json = json.load(f)

    # Immutable metadata: set explicitly, never sent to the LLM
    if isinstance(source_json, dict):
        if "culturecode" in source_json:
            source_json["culturecode"] = lang_code
        if "name" in source_json and native_name:
            source_json["name"] = native_name

    # Only collect strings outside the immutable top-level keys
    string_references = collect_translatable(source_json)

    # Unique translatable strings only, remembering each string's JSON key as context
    unique_texts = []
    key_hints = {}
    seen = set()
    for _, key_or_idx, text in string_references:
        if text not in seen and needs_translation(text):
            seen.add(text)
            unique_texts.append(text)
            if isinstance(key_or_idx, str):
                key_hints[text] = key_or_idx

    if args.resume:
        src_paths = collect_paths(source_json)
        cache, new_count, removed_count = load_existing_translations(output_file, src_paths)
        if cache:
            print(f" -> Updating existing '{output_file}': keeping {len(cache)} translations, "
                  f"{new_count} new strings, {removed_count} removed keys will be dropped.")
    else:
        cache = {}
    remaining = [t for t in unique_texts if t not in cache]
    print(f" -> {len(string_references)} strings, {len(unique_texts)} unique translatable, {len(remaining)} to do.")

    def write_output():
        for parent, key_or_idx, original in string_references:
            parent[key_or_idx] = cache.get(original, original)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(source_json, f, ensure_ascii=False, indent=2)

    failures = set()
    total_batches = (len(remaining) + args.chunk_size - 1) // args.chunk_size
    for i in range(0, len(remaining), args.chunk_size):
        chunk = remaining[i:i + args.chunk_size]
        print(f"   [Batch {i // args.chunk_size + 1}/{total_batches}] Translating {len(chunk)} strings...")
        translate_texts(chunk, english_name, model, cache, failures, key_hints)
        write_output()  # incremental save: safe to interrupt & resume

    write_output()
    if failures:
        print(f"💾 Saved '{output_file}'. ({len(failures)} strings failed and were kept in English — see ❌ warnings above.)")
    else:
        identical = sum(1 for t in unique_texts if cache.get(t) == t)
        print(f"💾 Saved '{output_file}'. All strings translated ({identical} came back identical to English, e.g. brand names/technical terms).")


def main():
    parser = argparse.ArgumentParser(description="Translate English JSON tokens via Ollama.")
    parser.add_argument("languages", nargs="+",
                        help="Target language codes (e.g. fr de zh-cn) or 'all' for every supported language")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", default=None, help="Output file (single language only; default <code>.json)")
    parser.add_argument("-m", "--model", default=None,
                        help="Force one model for all languages (default: auto-pick per language)")
    parser.add_argument("-c", "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Regenerate the whole file from scratch instead of updating the existing one")
    args = parser.parse_args()

    if args.languages == ["all"]:
        args.languages = ALL_LANGUAGES
    if args.output and len(args.languages) > 1:
        parser.error("--output can only be used with a single language")
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found.")
        sys.exit(1)

    # Ensure every model we'll need is installed (deduped)
    for model in sorted({pick_model(lc, args.model) for lc in args.languages}):
        ensure_model_installed(model)

    for lang_code in args.languages:
        translate_language(lang_code, args)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
