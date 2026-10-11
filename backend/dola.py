"""Dola (www.dola.com) web API client for Seedance video generation.

Flow (reverse-engineered from the Dola web client):
  1. GET https://www.dola.com/ -> public cookies (ttwid)
  2. Merge with the user's auth cookies (sessionid, ...)
  3. POST https://www.dola.com/chat/completion?... with chat_ability
     {ability_type: 17, ability_param: {model, duration, ratio}}
  4. Parse conversation_id from the SSE stream
  5. Poll POST .../im/chain/single -> video id (vid)
  6. Poll POST .../samantha/video/get_play_info {"vid": vid} -> mp4 URL
  7. Download the MP4
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

import requests

ANDROID_WEBVIEW_UA = (
    "Mozilla/5.0 (Linux; Android 14; SM-S928B Build/UP1A.231005.007; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/130.0.0.0 Mobile Safari/537.36"
)

VIDEO_POLL_ATTEMPTS = 240          # 240 x 5s = ~20 min max per video
VIDEO_POLL_INTERVAL = 5
PLAY_INFO_ATTEMPTS = 120
PLAY_INFO_INTERVAL = 5

VIDEO_FAILURE_MARKERS = (
    "failed to generate", "generation failed", "unable to generate",
    "cannot generate", "can't generate", "cant generate", "couldn't generate",
    "could not generate", "no points", "insufficient points", "high demand",
    "rate limit", "limit reached", "try again later", "may violate our policies",
    "violate our policies", "violates our policies", "policy violation",
    "modify it and try again", "input contains content", "safety policy",
    "content policy", "inappropriate", "system error",
)


class DolaError(RuntimeError):
    pass


class DolaTerminalError(DolaError):
    """Non-retryable: Dola rejected this specific prompt."""


@dataclass
class DolaSession:
    url: str
    headers: dict
    payload_template: dict
    fp: str
    has_auth_cookies: bool
    unwatermarked_url: str = ""
    seen_fallback_apis: set = field(default_factory=set)


# ---------------------------------------------------------------- cookies

def parse_cookie_text(text: str) -> dict:
    """Accept Netscape format, Cookie-Editor JSON, or raw header string."""
    text = (text or "").strip()
    if not text:
        return {}
    # JSON array (Cookie-Editor export)
    if text.lstrip().startswith("["):
        try:
            entries = json.loads(text[text.find("["):])
            return {e["name"]: e["value"] for e in entries if "name" in e}
        except Exception:
            pass
    # Netscape format
    if text.startswith("#") or "\t" in text:
        cookies = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") and "\t" not in line:
                continue
            if line.startswith("#HttpOnly_"):
                line = line[len("#HttpOnly_"):]
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]
        if cookies:
            return cookies
    # raw header string
    cookies = {}
    for part in text.split(";"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            cookies[k.strip()] = v.strip()
    return cookies


def format_cookie_header(cookies: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in cookies.items() if v)


# ---------------------------------------------------------------- payload

def generate_fp() -> str:
    h = uuid.uuid4().hex
    h2 = uuid.uuid4().hex
    return f"verify_{h[:8]}_{h[8:16]}_{h2[:4]}_{h2[4:8]}_{h2[8:12]}_{h2[12:24]}"


def base_payload(fp: str | None = None) -> dict:
    fp = fp or generate_fp()
    now_ms = int(time.time() * 1000)
    now_sec = int(time.time())
    return {
        "client_meta": {
            "local_conversation_id": f"local_{uuid.uuid4().int % 10000000000000000}",
            "conversation_id": "",
            "bot_id": "7339470689562525703",
            "last_section_id": "",
            "last_message_index": None,
        },
        "messages": [{
            "local_message_id": str(uuid.uuid4()),
            "content_block": [{
                "block_type": 10000,
                "content": {"text_block": {"text": "placeholder", "icon_url": "",
                                           "icon_url_dark": "", "summary": ""},
                            "pc_event_block": ""},
                "block_id": str(uuid.uuid4()),
                "parent_id": "",
                "meta_info": [],
                "append_fields": [],
            }],
            "message_status": 0,
        }],
        "option": {
            "send_message_scene": "", "create_time_ms": now_ms, "collect_id": "",
            "is_audio": False, "answer_with_suggest": False, "tts_switch": False,
            "need_deep_think": 0, "click_clear_context": False, "from_suggest": False,
            "is_regen": False, "is_replace": False, "is_from_click_option": False,
            "disable_sse_cache": False, "select_text_action": "", "is_select_text": False,
            "resend_for_regen": False, "scene_type": 0, "unique_key": str(uuid.uuid4()),
            "start_seq": 0, "need_create_conversation": True,
            "conversation_init_option": {"need_ack_conversation": True},
            "regen_query_id": [], "edit_query_id": [], "regen_instruction": "",
            "no_replace_for_regen": False, "message_from": 0, "shared_app_name": "",
            "shared_app_id": "",
            "sse_recv_event_options": {"support_chunk_delta": True},
            "is_ai_playground": False, "is_old_user": False,
            "recovery_option": {"is_recovery": False,
                                "req_create_time_sec": now_sec,
                                "append_sse_event_scene": 0},
            "message_storage_type": 0,
        },
        "user_context": [],
        "ext": {
            "use_deep_think": "0", "fp": fp, "sub_conv_firstmet_type": "1",
            "collection_id": "",
            "conversation_init_option": '{"need_ack_conversation":true}',
            "commerce_credit_config_enable": "0",
        },
    }


def normalize_model(model: str) -> str:
    m = (model or "seedance_v2.5").strip().lower().replace(" ", "_")
    m = {"seedance_2.0": "seedance_v2.0", "seedance_2.5": "seedance_v2.5",
         "seedance_v2": "seedance_v2.0"}.get(m, m)
    if m not in {"seedance_v2.0", "seedance_v2.5"}:
        raise DolaError(f"Unsupported model: {model}")
    return m


def build_payload(template: dict, prompt: str, duration: int, ratio: str,
                  model: str = "seedance_v2.5") -> dict:
    payload = json.loads(json.dumps(template))
    model = normalize_model(model)
    clean = re.sub(r"(?i)^generate\s+video\s*:\s*", "", prompt.strip()).strip()
    payload["messages"][0]["local_message_id"] = str(uuid.uuid4())
    payload["messages"][0]["content_block"][0]["block_id"] = str(uuid.uuid4())
    payload["messages"][0]["content_block"][0]["content"]["text_block"]["text"] = \
        f"Generate video: {clean}"
    payload["option"]["unique_key"] = str(uuid.uuid4())
    payload["option"]["create_time_ms"] = int(time.time() * 1000)
    payload["option"]["recovery_option"]["req_create_time_sec"] = int(time.time())
    payload["chat_ability"] = {
        "ability_type": 17,
        "ability_param": json.dumps({"model": model, "duration": int(duration),
                                     "ratio": ratio or "9:16"},
                                    separators=(",", ":")),
    }
    payload["client_meta"]["conversation_id"] = ""
    payload["client_meta"]["last_section_id"] = ""
    payload["client_meta"]["last_message_index"] = None
    payload["client_meta"]["local_conversation_id"] = \
        f"local_{uuid.uuid4().int % 10000000000000000}"
    return payload


# ---------------------------------------------------------------- parsing

def parse_conversation_from_stream(text: str) -> tuple[str, int]:
    m = re.search(r'"conversation_id"\s*:\s*"([0-9]+)"', text)
    if not m:
        em = re.search(r'"(?:error_msg|message|msg)"\s*:\s*"([^"]+)"', text)
        if em:
            raise DolaError(f"Dola error: {em.group(1)}")
        raise DolaError(f"No conversation_id in response: {text[:300]}")
    types = re.findall(r'"conversation_type"\s*:\s*([0-9]+)', text)
    return m.group(1), int(types[-1]) if types else 3


def _find_vid_recursive(value) -> str | None:
    if isinstance(value, str):
        m = re.search(r'"vid"\s*:\s*"([a-zA-Z0-9_:-]+)"', value)
        if m:
            return m.group(1)
        try:
            return _find_vid_recursive(json.loads(value))
        except Exception:
            return None
    if isinstance(value, list):
        for item in value:
            v = _find_vid_recursive(item)
            if v:
                return v
    if isinstance(value, dict):
        vv = value.get("vid")
        if isinstance(vv, str) and vv:
            return vv
        for child in value.values():
            v = _find_vid_recursive(child)
            if v:
                return v
    return None


def parse_vid(payload: dict) -> str | None:
    if payload.get("code", 0) != 0:
        raise DolaError(f"Dola API error {payload.get('code')}: {payload.get('message')}")
    m = re.search(r'"vid"\s*:\s*"([a-zA-Z0-9_:-]+)"', json.dumps(payload))
    if m:
        return m.group(1)
    return _find_vid_recursive(payload)


def _extract_texts(value, out: list):
    if isinstance(value, str):
        s = value.strip()
        if s:
            try:
                _extract_texts(json.loads(s), out)
            except Exception:
                out.append(s)
    elif isinstance(value, list):
        for i in value:
            _extract_texts(i, out)
    elif isinstance(value, dict):
        for k in ("text", "summary", "message", "msg"):
            if isinstance(value.get(k), str):
                out.append(value[k])
        tb = value.get("text_block")
        if isinstance(tb, dict) and isinstance(tb.get("text"), str):
            out.append(tb["text"])
        for k in ("content", "data", "delta", "answer"):
            if k in value:
                _extract_texts(value[k], out)


def extract_chain_texts(payload: dict) -> list[str]:
    found: list[dict] = []

    def walk(v):
        if isinstance(v, dict):
            msgs = v.get("messages")
            if isinstance(msgs, list):
                found.extend(m for m in msgs if isinstance(m, dict))
            for c in v.values():
                walk(c)
        elif isinstance(v, list):
            for c in v:
                walk(c)

    walk(payload)
    texts, seen = [], set()
    for msg in found:
        raw: list = []
        _extract_texts(msg.get("content") or msg.get("message") or msg.get("text") or "", raw)
        for t in raw:
            c = " ".join(t.split())
            if c and c not in seen:
                seen.add(c)
                texts.append(c)
    return texts


def is_terminal_failure(text: str) -> bool:
    low = text.lower().replace("’", "'")
    return any(m in low for m in VIDEO_FAILURE_MARKERS)


def parse_play_info(payload: dict) -> str | None:
    """Extract best mp4 URL from get_play_info response."""
    urls: list[str] = []

    def walk(v):
        if isinstance(v, str):
            if v.startswith("http") and (".mp4" in v or "video" in v):
                urls.append(v)
        elif isinstance(v, dict):
            for c in v.values():
                walk(c)
        elif isinstance(v, list):
            for c in v:
                walk(c)

    walk(payload)
    # prefer URLs that look like direct mp4 streams
    for u in urls:
        if ".mp4" in u:
            return u
    return urls[0] if urls else None


# ---------------------------------------------------------------- client

class DolaClient:
    def __init__(self, timeout: int = 60, proxy: str | None = None):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ANDROID_WEBVIEW_UA})
        if proxy:
            self.s.proxies = {"http": proxy, "https": proxy}
        self.timeout = timeout

    def build_session(self, auth_cookies: str, region: str = "US") -> DolaSession:
        # 1. public cookies
        try:
            r = self.s.get("https://www.dola.com/", timeout=20,
                           headers={"User-Agent": ANDROID_WEBVIEW_UA})
            public = {c.name: c.value for c in self.s.cookies}
        except Exception as exc:
            raise DolaError(f"Cannot reach dola.com: {exc}")
        if not public.get("ttwid"):
            # try once more; some regions need it
            raise DolaError("No ttwid cookie from dola.com (site may be blocked here)")

        auth = parse_cookie_text(auth_cookies)
        merged = {"i18next": "en", "flow_user_country": region}
        merged.update(auth)
        for k, v in public.items():
            merged.setdefault(k, v)

        device_id = str(uuid.uuid4().int)[:19]
        tea_uuid = str(uuid.uuid4().int)[:19]
        web_tab_id = str(uuid.uuid4())
        fp = generate_fp()
        url = (
            "https://www.dola.com/chat/completion"
            f"?aid=495671&device_id={device_id}&device_platform=android&fp={fp}"
            f"&language=en&pc_version=3.23.5&pkg_type=release_version&real_aid=495671"
            f"&region={region}&samantha_web=1&sys_region={region}&tea_uuid={tea_uuid}"
            f"&use-olympus-account=1&version_code=20800&web_id={tea_uuid}"
            f"&web_platform=web&web_tab_id={web_tab_id}"
        )
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "agw-js-conv": "str, str",
            "content-type": "application/json",
            "cookie": format_cookie_header(merged),
            "origin": "https://www.dola.com",
            "referer": "https://www.dola.com/chat/",
            "User-Agent": ANDROID_WEBVIEW_UA,
            "sec-ch-ua": '"Chromium";v="130", "Android WebView";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "x-requested-with": "com.android.browser",
        }
        return DolaSession(url=url, headers=headers,
                           payload_template=base_payload(fp), fp=fp,
                           has_auth_cookies=bool(auth.get("sessionid")))

    def submit_video(self, session: DolaSession, prompt: str, duration: int,
                     ratio: str, model: str = "seedance_v2.5",
                     log=None) -> tuple[str, int]:
        payload = build_payload(session.payload_template, prompt, duration, ratio, model)
        r = self.s.post(session.url, headers=session.headers, json=payload,
                        timeout=self.timeout)
        if r.status_code in (401, 403):
            raise DolaError(f"Auth rejected (HTTP {r.status_code}) — cookies expired?")
        if r.status_code == 429:
            raise DolaError("Rate limited (HTTP 429) — account resting needed")
        if r.status_code != 200:
            raise DolaError(f"Submit failed: HTTP {r.status_code}: {r.text[:200]}")
        cid, ctype = parse_conversation_from_stream(r.text)
        if log:
            log(f"Submitted -> conversation {cid[-8:]}")
        return cid, ctype

    def poll_video_id(self, session: DolaSession, conversation_id: str,
                      conversation_type: int, log=None,
                      cancel=None) -> str | None:
        url = session.url.replace("chat/completion", "im/chain/single")
        headers = {k: v for k, v in session.headers.items()
                   if k.lower() not in ("content-type", "agw-js-conv")}
        headers["content-type"] = "application/json; encoding=utf-8"
        headers["agw-js-conv"] = "str"
        seen = set()
        for attempt in range(1, VIDEO_POLL_ATTEMPTS + 1):
            if cancel and cancel():
                return None
            body = {
                "cmd": 3100,
                "uplink_body": {"pull_singe_chain_uplink_body": {
                    "conversation_id": conversation_id, "anchor_index": 0,
                    "conversation_type": conversation_type, "direction": 3,
                    "limit": 50, "ext": {}, "filter": {"index_list": []},
                    "evaluate_ab_params": "", "evaluate_common_params": ""}},
                "sequence_id": str(uuid.uuid4()), "channel": 2, "version": "1",
            }
            try:
                r = self.s.post(url, headers=headers, json=body, timeout=self.timeout)
            except Exception as exc:
                if log:
                    log(f"poll error: {exc}")
                time.sleep(VIDEO_POLL_INTERVAL)
                continue
            if r.status_code == 200:
                try:
                    payload = r.json()
                except Exception:
                    time.sleep(VIDEO_POLL_INTERVAL)
                    continue
                vid = parse_vid(payload)
                if vid:
                    if log:
                        log("Video id received ✓")
                    return vid
                for msg in extract_chain_texts(payload):
                    if msg not in seen:
                        seen.add(msg)
                        if log:
                            log(msg[:220])
                        if is_terminal_failure(msg):
                            raise DolaTerminalError(f"Dola: {msg[:220]}")
            time.sleep(VIDEO_POLL_INTERVAL)
        return None

    def poll_download_url(self, session: DolaSession, vid: str,
                          log=None, cancel=None) -> str | None:
        if session.unwatermarked_url:
            return session.unwatermarked_url
        url = session.url.replace("chat/completion", "samantha/video/get_play_info")
        headers = {k: v for k, v in session.headers.items()
                   if k.lower() != "content-type"}
        headers["content-type"] = "application/json"
        for _ in range(PLAY_INFO_ATTEMPTS):
            if cancel and cancel():
                return None
            try:
                r = self.s.post(url, headers=headers, json={"vid": vid},
                                timeout=self.timeout)
                if r.status_code == 200:
                    dl = parse_play_info(r.json())
                    if dl:
                        if log:
                            log("Download URL ready ✓")
                        return dl
            except Exception as exc:
                if log:
                    log(f"play_info error: {exc}")
            time.sleep(PLAY_INFO_INTERVAL)
        return None

    def download(self, url: str, path: str, log=None):
        with self.s.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    f.write(chunk)
                    done += len(chunk)
            if log:
                log(f"Saved {done/1048576:.1f} MB")

    def check_health(self, auth_cookies: str, region: str = "US") -> dict:
        """Lightweight check: can we build a session + do auth cookies exist?"""
        try:
            session = self.build_session(auth_cookies, region)
            # verify auth by hitting a lightweight authed endpoint
            r = self.s.post(session.url.replace("chat/completion", "im/chain/single"),
                            headers={**session.headers,
                                     "content-type": "application/json; encoding=utf-8"},
                            json={"cmd": 3100,
                                  "uplink_body": {"pull_singe_chain_uplink_body": {
                                      "conversation_id": "0", "anchor_index": 0,
                                      "conversation_type": 3, "direction": 3, "limit": 1,
                                      "ext": {}, "filter": {"index_list": []},
                                      "evaluate_ab_params": "", "evaluate_common_params": ""}},
                                  "sequence_id": str(uuid.uuid4()),
                                  "channel": 2, "version": "1"},
                            timeout=30)
            ok = r.status_code == 200
            return {"ok": ok, "has_auth": session.has_auth_cookies,
                    "detail": f"HTTP {r.status_code}" if not ok else "session live"}
        except DolaError as exc:
            return {"ok": False, "has_auth": False, "detail": str(exc)[:160]}
        except Exception as exc:
            return {"ok": False, "has_auth": False,
                    "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}
