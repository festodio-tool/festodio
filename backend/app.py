"""Festodio — Seedance 2.5 batch video automation (web tool)."""
from __future__ import annotations

import os
import secrets
import threading
from functools import wraps

from flask import Flask, jsonify, request, send_file, send_from_directory, session

from engine import engine, VIDEOS_DIR, DATA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE, "frontend")
SKILLS = os.path.join(BASE, "skills")

app = Flask(__name__)
app.secret_key = os.environ.get("FESTODIO_SECRET", secrets.token_hex(32))
ADMIN_PASSWORD = os.environ.get("FESTODIO_PASSWORD", "festodio123")


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get("auth"):
            return jsonify({"ok": False, "msg": "login required"}), 401
        return f(*a, **kw)
    return wrapper


# ------------------------------------------------------------ pages
@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    if data.get("password") == ADMIN_PASSWORD:
        session["auth"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "wrong password"})


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("auth", None)
    return jsonify({"ok": True})


# ------------------------------------------------------------ status
@app.route("/api/status")
@login_required
def status():
    q = engine.queue_state()["counts"]
    accs = engine.list_accounts()
    live = sum(1 for a in accs if a["status"] == "live")
    return jsonify({
        "running": engine.running,
        "queue": q,
        "accounts_total": len(accs),
        "accounts_live": live,
        "settings": engine.settings,
    })


# ------------------------------------------------------------ prompts / batch
@app.route("/api/prompts", methods=["POST"])
@login_required
def add_prompts():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    prompts = [p.strip() for p in text.split("\n") if p.strip()]
    n = engine.add_prompts(prompts)
    return jsonify({"ok": True, "queued": n})


@app.route("/api/prompts/file", methods=["POST"])
@login_required
def upload_prompts():
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "msg": "no file"})
    text = f.read().decode("utf-8", errors="ignore")
    prompts = [p.strip() for p in text.split("\n") if p.strip()]
    n = engine.add_prompts(prompts)
    return jsonify({"ok": True, "queued": n})


@app.route("/api/batch/start", methods=["POST"])
@login_required
def batch_start():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(engine.start(data))


@app.route("/api/batch/stop", methods=["POST"])
@login_required
def batch_stop():
    return jsonify(engine.stop())


@app.route("/api/queue")
@login_required
def queue():
    return jsonify(engine.queue_state())


@app.route("/api/jobs/clear", methods=["POST"])
@login_required
def jobs_clear():
    data = request.get_json(force=True, silent=True) or {}
    engine.clear_jobs(data.get("which", "all"))
    return jsonify({"ok": True})


# ------------------------------------------------------------ accounts
@app.route("/api/accounts")
@login_required
def accounts():
    return jsonify(engine.list_accounts())


@app.route("/api/accounts", methods=["POST"])
@login_required
def account_add():
    data = request.get_json(force=True, silent=True) or {}
    try:
        acc = engine.add_account(data.get("name", ""),
                                 data.get("cookies", ""),
                                 data.get("region", "US"))
        return jsonify({"ok": True, "account": acc})
    except ValueError as exc:
        return jsonify({"ok": False, "msg": str(exc)})


@app.route("/api/accounts/<acc_id>", methods=["DELETE"])
@login_required
def account_del(acc_id):
    engine.remove_account(acc_id)
    return jsonify({"ok": True})


@app.route("/api/accounts/<acc_id>/check", methods=["POST"])
@login_required
def account_check(acc_id):
    threading.Thread(target=engine.check_account,
                     args=(acc_id,), daemon=True).start()
    return jsonify({"ok": True, "msg": "checking..."})


@app.route("/api/accounts/check-all", methods=["POST"])
@login_required
def account_check_all():
    threading.Thread(target=engine.check_all_accounts, daemon=True).start()
    return jsonify({"ok": True, "msg": "checking all..."})


# ------------------------------------------------------------ skill file
@app.route("/api/skill/download")
@login_required
def skill_download():
    return send_file(
        os.path.join(SKILLS, "kartar-seedance-30s-SKILL.md"),
        as_attachment=True,
        download_name="kartar-seedance-30s-SKILL.md")


# ------------------------------------------------------------ logs / videos
@app.route("/api/logs")
@login_required
def logs():
    with engine.lock:
        return jsonify(list(engine.logs[-150:]))


@app.route("/api/videos")
@login_required
def videos():
    files = []
    for f in sorted(os.listdir(VIDEOS_DIR), reverse=True):
        if f.endswith(".mp4"):
            fp = os.path.join(VIDEOS_DIR, f)
            files.append({"name": f,
                          "mb": round(os.path.getsize(fp) / 1048576, 1)})
    return jsonify(files)


@app.route("/videos/<name>")
@login_required
def serve_video(name):
    return send_file(os.path.join(VIDEOS_DIR, name))


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
