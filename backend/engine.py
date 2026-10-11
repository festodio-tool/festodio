"""Batch generation engine: accounts, job queue, parallel workers."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dola import DolaClient, DolaError, DolaTerminalError, parse_cookie_text

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
ACCOUNTS_FILE = os.path.join(DATA, "accounts.json")
JOBS_FILE = os.path.join(DATA, "jobs.json")
LOG_FILE = os.path.join(DATA, "engine.log")
VIDEOS_DIR = os.path.join(DATA, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)


def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.accounts: list[dict] = _load_json(ACCOUNTS_FILE, [])
        self.jobs: list[dict] = _load_json(JOBS_FILE, [])
        self.logs: list[dict] = []
        self.running = False
        self.executor: ThreadPoolExecutor | None = None
        self.settings = {"duration": 30, "ratio": "9:16", "model": "seedance_v2.5",
                         "parallel": 3, "videos_per_prompt": 1, "region": "US",
                         "proxy": ""}
        self._stop_event = threading.Event()

    # ---------------------------------------------------------- logging
    def log(self, msg: str, level: str = "info", job_id: str | None = None):
        entry = {"t": datetime.now().strftime("%H:%M:%S"), "msg": msg,
                 "level": level, "job": job_id}
        with self.lock:
            self.logs.append(entry)
            self.logs = self.logs[-500:]
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{entry['t']} [{level}] {msg}\n")
        except Exception:
            pass

    # ---------------------------------------------------------- accounts
    def _save_accounts(self):
        _save_json(ACCOUNTS_FILE, self.accounts)

    def list_accounts(self):
        with self.lock:
            return [{"id": a["id"], "name": a["name"], "status": a.get("status", "?"),
                     "detail": a.get("detail", ""), "uses": a.get("uses", 0),
                     "region": a.get("region", "US")}
                    for a in self.accounts]

    def add_account(self, name: str, cookie_text: str, region: str = "US") -> dict:
        cookies = parse_cookie_text(cookie_text)
        if not cookies.get("sessionid"):
            raise ValueError("No sessionid found in cookies — login first, then export again")
        acc = {"id": uuid.uuid4().hex[:8], "name": name or f"Account {len(self.accounts)+1}",
               "cookies": cookie_text, "region": region or "US",
               "status": "?", "detail": "not checked yet", "uses": 0,
               "added": datetime.now().strftime("%Y-%m-%d %H:%M")}
        with self.lock:
            self.accounts.append(acc)
            self._save_accounts()
        self.log(f"Account added: {acc['name']}")
        return {"id": acc["id"], "name": acc["name"]}

    def remove_account(self, acc_id: str):
        with self.lock:
            self.accounts = [a for a in self.accounts if a["id"] != acc_id]
            self._save_accounts()
        self.log(f"Account removed: {acc_id}")

    def check_account(self, acc_id: str) -> dict:
        with self.lock:
            acc = next((a for a in self.accounts if a["id"] == acc_id), None)
        if not acc:
            return {"ok": False, "detail": "not found"}
        client = DolaClient(proxy=self.settings.get("proxy") or None)
        res = client.check_health(acc["cookies"], acc.get("region", "US"))
        with self.lock:
            acc["status"] = "live" if res["ok"] else "dead"
            acc["detail"] = res["detail"]
            self._save_accounts()
        self.log(f"Health {acc['name']}: {acc['status']} ({acc['detail']})",
                 "ok" if res["ok"] else "warn")
        return res

    def check_all_accounts(self):
        for a in self.list_accounts():
            self.check_account(a["id"])

    def live_accounts(self):
        with self.lock:
            return [a for a in self.accounts if a.get("status") == "live"]

    # ---------------------------------------------------------- jobs
    def _save_jobs(self):
        _save_json(JOBS_FILE, self.jobs)

    def add_prompts(self, prompts: list[str]) -> int:
        n = 0
        with self.lock:
            for p in prompts:
                p = (p or "").strip()
                if not p:
                    continue
                for _ in range(self.settings.get("videos_per_prompt", 1)):
                    self.jobs.append({"id": uuid.uuid4().hex[:8], "prompt": p,
                                      "status": "queued", "account": "", "file": "",
                                      "error": "", "tries": 0,
                                      "created": datetime.now().strftime("%m-%d %H:%M")})
                    n += 1
            self._save_jobs()
        self.log(f"{n} job(s) queued")
        return n

    def clear_jobs(self, which="all"):
        with self.lock:
            if which == "all":
                self.jobs = []
            else:
                self.jobs = [j for j in self.jobs if j["status"] not in
                             (["failed", "done"] if which == "done" else [which])]
            self._save_jobs()

    def queue_state(self):
        with self.lock:
            counts = {"queued": 0, "active": 0, "done": 0, "failed": 0}
            for j in self.jobs:
                counts[j["status"]] = counts.get(j["status"], 0) + 1
            return {"counts": counts,
                    "jobs": list(reversed(self.jobs[-80:]))}

    # ---------------------------------------------------------- batch run
    def start(self, settings: dict | None = None):
        if self.running:
            return {"ok": False, "msg": "already running"}
        if settings:
            self.settings.update(settings)
        live = self.live_accounts()
        if not live:
            return {"ok": False, "msg": "No live accounts — add cookies & check health first"}
        with self.lock:
            pending = [j for j in self.jobs if j["status"] in ("queued", "failed")]
            for j in pending:
                if j["status"] == "failed":
                    j["status"] = "queued"
                    j["error"] = ""
            self._save_jobs()
        if not pending:
            return {"ok": False, "msg": "Queue is empty — add prompts first"}
        self.running = True
        self._stop_event.clear()
        n = max(1, min(8, int(self.settings.get("parallel", 3))))
        self.executor = ThreadPoolExecutor(max_workers=n,
                                           thread_name_prefix="dola-worker")
        for job in pending:
            self.executor.submit(self._run_job, job["id"])
        self.log(f"Batch started: {len(pending)} jobs, {n} workers, "
                 f"{self.settings['duration']}s {self.settings['ratio']}")
        return {"ok": True, "msg": f"started {len(pending)} jobs"}

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        with self.lock:
            for j in self.jobs:
                if j["status"] == "active":
                    j["status"] = "queued"
            self._save_jobs()
        self.log("Batch stopped", "warn")
        return {"ok": True}

    def _pick_account(self):
        with self.lock:
            live = [a for a in self.accounts if a.get("status") == "live"]
            if not live:
                return None
            live.sort(key=lambda a: a.get("uses", 0))
            acc = live[0]
            acc["uses"] = acc.get("uses", 0) + 1
            self._save_accounts()
            return dict(acc)

    def _set_job(self, job_id, **kw):
        with self.lock:
            for j in self.jobs:
                if j["id"] == job_id:
                    j.update(kw)
                    break
            self._save_jobs()

    @staticmethod
    def _is_transient(msg: str) -> bool:
        m = msg.lower()
        return any(k in m for k in (
            "high demand", "try again later", "too many requests",
            "rate limit", "429", "502", "503", "504", "timeout",
            "timed out", "connection", "network", "service unavailable",
            "bad gateway", "unknownhost", "sockettimeout"))

    def _run_job(self, job_id: str):
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
            if not job or job["status"] not in ("queued",):
                return
            job["status"] = "active"
            self._save_jobs()
            prompt = job["prompt"]

        acc = self._pick_account()
        if not acc:
            self._set_job(job_id, status="queued")
            self.log("No live account available — job re-queued", "warn", job_id)
            return

        def log(m, level="info"):
            self.log(f"[{acc['name']}] {m}", level, job_id)

        self._set_job(job_id, account=acc["name"],
                      tries=(job.get("tries", 0) + 1))
        log(f"Working: {prompt[:60]}...")

        # Retry loop for transient errors (high demand, rate limits, network)
        max_attempts = 6
        backoffs = [0, 45, 120, 300, 600, 900]  # seconds
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            if self._stop_event.is_set():
                break
            if attempt > 1:
                wait = backoffs[min(attempt - 1, len(backoffs) - 1)]
                log(f"Retry {attempt}/{max_attempts} in {wait}s (server busy)...", "warn")
                if self._stop_event.wait(wait):
                    break
            try:
                client = DolaClient(proxy=self.settings.get("proxy") or None)
                session = client.build_session(acc["cookies"], acc.get("region", "US"))
                cid, ctype = client.submit_video(
                    session, prompt, self.settings["duration"],
                    self.settings["ratio"], self.settings["model"], log=log)
                vid = client.poll_video_id(session, cid, ctype, log=log,
                                           cancel=self._stop_event.is_set)
                if not vid:
                    raise DolaError("Timed out waiting for video")
                url = client.poll_download_url(session, vid, log=log,
                                               cancel=self._stop_event.is_set)
                if not url:
                    raise DolaError("Timed out waiting for download URL")
                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}.mp4"
                fpath = os.path.join(VIDEOS_DIR, fname)
                client.download(url, fpath, log=log)
                self._set_job(job_id, status="done", file=fname)
                log(f"Done ✓ {fname}", "ok")
                return
            except DolaTerminalError as exc:
                last_error = str(exc)[:200]
                break  # terminal — do not retry
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:150]}"
                ml = last_error.lower()
                if "expired" in ml or "auth rejected" in ml:
                    with self.lock:
                        for a in self.accounts:
                            if a["id"] == acc["id"]:
                                a["status"] = "dead"
                                a["detail"] = "cookies expired?"
                        self._save_accounts()
                    break  # auth dead — do not retry
                if not self._is_transient(last_error):
                    break  # permanent — do not retry
                log(f"Transient error (attempt {attempt}/{max_attempts}): "
                    f"{last_error[:120]}", "warn")
                # loop continues -> backoff + retry

        self._set_job(job_id, status="failed", error=last_error[:200])
        log(f"Failed after attempts: {last_error[:150]}", "err")


engine = Engine()
