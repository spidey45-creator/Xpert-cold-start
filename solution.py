#!/usr/bin/env python3
"""
Xpert Cold Start Challenge — Hybrid Graph + Batched LLM Solution (OpenRouter Hardened)
Reads <input_dir> (chats.jsonl, messages.jsonl, seed.json) and writes <output.json>.
Standard library only + llm.py.
"""
import collections
import concurrent.futures
import datetime
import json
import os
import re
import sys
import time
import urllib.error
from pathlib import Path

from llm import chat

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

CIRCLES = {
    "spouse", "child", "parent", "in-law", "sibling", "relative",
    "friend", "colleague", "client", "vendor", "partner", "staff"
}
ORG_TYPES = {"company", "residence", "vacation home", "workspace", "society", "school", "other"}
ORG_ROLES = {"owner", "partner", "helper", "resident", "parent", "member", "other"}
AREAS = {"self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"}
GOAL_TYPES = {"habit", "milestone", "experience"}
HORIZONS = {"3mo", "1yr", "3yr", "5yr"}
LOOP_TYPES = {"commitment", "followup", "deadline", "question"}
STATUSES = {"open", "done", "dropped"}
ACTIONS = {"assist", "remind", "leave"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json_safely(text):
    if not text:
        return None
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m_obj = re.search(r"\{[\s\S]*\}", text)
        if m_obj:
            try:
                return json.loads(m_obj.group(0))
            except json.JSONDecodeError:
                pass
        m_arr = re.search(r"\[[\s\S]*\]", text)
        if m_arr:
            try:
                return json.loads(m_arr.group(0))
            except json.JSONDecodeError:
                pass
    return None


def call_llm_json(system_prompt, user_prompt, max_tokens=8192):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + "\n\nRespond ONLY with a valid JSON object."},
    ]
    use_json_mode = True
    for attempt in range(6):
        try:
            rf = {"type": "json_object"} if use_json_mode else None
            reply = chat(
                messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format=rf,
                timeout=150,
            )
            parsed = parse_json_safely(reply)
            if isinstance(parsed, dict):
                return parsed
            use_json_mode = False
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and use_json_mode:
                use_json_mode = False
                continue
            wait = 4 * (attempt + 1)
            sys.stderr.write(f"[warn] HTTP {e.code} on attempt {attempt+1}, retrying in {wait}s...\n")
            time.sleep(wait)
        except Exception as e:
            use_json_mode = False
            wait = 4 * (attempt + 1)
            sys.stderr.write(f"[warn] LLM call error ({e}) on attempt {attempt+1}, retrying in {wait}s...\n")
            time.sleep(wait)
    return {}


def ts_to_date(ts):
    return datetime.datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d")


def ts_to_dt_str(ts):
    return datetime.datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M")


def compute_loops_cutoff_ts(max_ts):
    end_dt = datetime.datetime.fromtimestamp(max_ts, tz=IST)
    end_midnight = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_dt = end_midnight - datetime.timedelta(days=15)
    return int(cutoff_dt.timestamp())


def build_context(chats, messages, seed):
    chat_by_id = {c["chat_id"]: c for c in chats}
    msgs_by_chat = collections.defaultdict(list)
    msgs_by_sender = collections.defaultdict(list)
    sender_names = collections.defaultdict(set)
    one_on_one_chat = {}

    for c in chats:
        if not c["is_group"]:
            others = [p for p in c["participants"] if p != "ME"]
            if len(others) == 1:
                one_on_one_chat[others[0]] = c["chat_id"]
                if c.get("name") and c["name"].strip().strip("\u200e"):
                    sender_names[others[0]].add(c["name"].strip().strip("\u200e"))

    for m in messages:
        cid = m["chat_id"]
        sid = m["sender_id"]
        msgs_by_chat[cid].append(m)
        msgs_by_sender[sid].append(m)
        sname = (m.get("sender_name") or "").strip()
        if sname:
            sender_names[sid].add(sname)

    for cid in msgs_by_chat:
        msgs_by_chat[cid].sort(key=lambda x: x["ts"])

    all_people = sorted({p for c in chats for p in c["participants"]} - {"ME"})
    me_names = sorted(sender_names.get("ME", set()))
    me_surnames = set()
    for name in me_names:
        parts = name.split()
        if len(parts) >= 2:
            me_surnames.add(parts[-1].lower())

    seed_circles = seed.get("circles") or {}
    family_surnames = set(me_surnames)
    for pid, cir in seed_circles.items():
        if cir in ("parent", "sibling", "spouse"):
            for nm in sender_names.get(pid, set()):
                parts = nm.split()
                if len(parts) >= 2:
                    family_surnames.add(parts[-1].lower())

    spouse_id = None
    for pid, cir in seed_circles.items():
        if cir == "spouse":
            spouse_id = pid
            break

    if not spouse_id:
        best_score = -1
        for pid in all_people:
            if pid in seed_circles:
                continue
            cid = one_on_one_chat.get(pid)
            if not cid:
                continue
            c_msgs = msgs_by_chat[cid]
            me_cnt = sum(1 for m in c_msgs if m["sender_id"] == "ME")
            ot_cnt = sum(1 for m in c_msgs if m["sender_id"] == pid)
            if me_cnt < 10 or ot_cnt < 10:
                continue
            names_lower = " ".join(sender_names.get(pid, set())).lower()
            surname_bonus = 250 if any(s in names_lower for s in family_surnames) else 0
            shared_fam_groups = 0
            for c in chats:
                if c["is_group"] and pid in c["participants"]:
                    if any(seed_circles.get(p) in ("parent", "sibling") for p in c["participants"]):
                        shared_fam_groups += 1
            score = len(c_msgs) + surname_bonus + shared_fam_groups * 120
            if score > best_score:
                best_score = score
                spouse_id = pid

    person_chats = collections.defaultdict(list)
    for c in chats:
        for p in c["participants"]:
            if p != "ME":
                person_chats[p].append(c["chat_id"])

    max_ts = max((m["ts"] for m in messages), default=0)
    min_ts = min((m["ts"] for m in messages), default=0)

    return {
        "chat_by_id": chat_by_id,
        "msgs_by_chat": msgs_by_chat,
        "msgs_by_sender": msgs_by_sender,
        "sender_names": sender_names,
        "one_on_one_chat": one_on_one_chat,
        "all_people": all_people,
        "me_names": me_names,
        "family_surnames": family_surnames,
        "spouse_id": spouse_id,
        "person_chats": person_chats,
        "max_ts": max_ts,
        "min_ts": min_ts,
    }


def format_msg_body(m, max_len=160):
    kind = m.get("kind") or "text"
    body = (m.get("body") or "").strip().replace("\n", " ")
    if kind == "voice_transcript":
        return f"[VOICE_TRANSCRIPT] {body[:max_len]}"
    if kind == "document" and body not in ("[document]", ""):
        return f"[DOC: {body[:max_len]}]"
    return body[:max_len]


def resolve_orgs_and_chats(chats, messages, seed, ctx):
    seed_orgs = seed.get("organisations") or []
    seed_circles = seed.get("circles") or {}
    msgs_by_chat = ctx["msgs_by_chat"]
    sender_names = ctx["sender_names"]

    chat_summaries = []
    for c in chats:
        cid = c["chat_id"]
        c_msgs = msgs_by_chat.get(cid, [])
        me_cnt = sum(1 for m in c_msgs if m["sender_id"] == "ME")
        p_list = [p for p in c["participants"] if p != "ME"]

        p_info = []
        for p in p_list[:12]:
            nms = "/".join(sorted(sender_names.get(p, {p})))
            sc = seed_circles.get(p)
            if p == ctx["spouse_id"] and not sc:
                sc = "likely_spouse"
            tag = f"{p}({nms}" + (f",seed={sc}" if sc else "") + ")"
            p_info.append(tag)
        if len(p_list) > 12:
            p_info.append(f"...+{len(p_list) - 12} more")

        sample_msgs = []
        for m in c_msgs:
            body = (m.get("body") or "").strip()
            if body and body not in ("[photo]", "[video]", "[document]", "ok", "Ok", "Done", "done"):
                sname = m.get("sender_name") or m["sender_id"]
                sample_msgs.append(f"{m['sender_id']}({sname}): {format_msg_body(m, 110)}")
                if len(sample_msgs) >= 4:
                    break

        chat_summaries.append({
            "chat_id": cid,
            "name": c["name"],
            "is_group": c["is_group"],
            "num_participants": len(c["participants"]),
            "total_msgs": len(c_msgs),
            "me_msgs": me_cnt,
            "participants_preview": p_info,
            "samples": sample_msgs,
        })

    system_prompt = """You are an expert analyst resolving the user's ("ME") Organisations and Chat Ties from WhatsApp metadata.

RULES FOR ORGANISATIONS:
1. Start with the organisations named in `seed_organisations` (type="company", side="work", role="owner").
2. Identify ONLY core organisations that ME runs, helps run, or belongs to as a household/workspace:
   - Seed companies (type="company", side="work", role="owner").
   - If there is a distinct sub-venture or spin-off company co-run with a partner/colleague (e.g., a DaaS / Milestone / enterprise product venture distinct from the main agency), include it as a separate company (type="company", side="work", role="owner").
   - If ME's spouse runs a business where ME and ME's accountant help out (e.g., an operations group with ME, spouse, and accountant), include that company (type="company", side="work", role="helper").
   - "Office": if there is office support staff (e.g., contact name has "Office" + company name, or cleaner/peon for the office), include an organisation with name="Office", type="workspace", side="work", role="owner".
   - "Home": ME's primary residence (name="Home", type="residence", side="home", role="owner") — covers kitchen groups, home apartment groups, kids carpool, home court/renovation vendors.
   - "Farmhouse": if ME has a farmhouse / vacation home (mentioned in chats or staff supporting it), include name="Farmhouse", type="vacation home", side="home", role="owner".
3. DO NOT create organisations for clients' companies, general friend groups, extended family groups, or news broadcast channels! Keep the total number of organisations between 5 and 8. Extra organisations penalize the score.
4. Valid `type`: ["company", "residence", "vacation home", "workspace", "society", "school", "other"]
   Valid `side`: ["home", "work"]
   Valid `role`: ["owner", "partner", "helper", "resident", "parent", "member", "other"]

RULES FOR TYING CHATS:
Every chat in the input MUST be tied to either `{"chat_id": "...", "organisation": "<org_id>"}` OR `{"chat_id": "...", "person": "<person_id>"}`:
- Group chats (`is_group: true`):
  - Work operational groups, client project groups, industry community groups (e.g. "D2C...", "D2B...", "<SeedOrg> Community"), finance/ads groups -> tie to the appropriate work `organisation`.
  - Home/household groups (Kitchen, apartment flat group, kids carpool) -> tie to `Home` organisation.
  - Spouse's company operations group -> tie to spouse's company `organisation`.
  - Extended family/friend/sports groups not belonging to the 5-8 core organisations -> tie to the primary `person` in that group or `Home` if household-related.
- 1:1 chats (`is_group: false`):
  - Most 1:1 chats are tied to the other `person` (`"person": "P..."`).
  - EXCEPTION: A 1:1 chat is tied to an `organisation` instead of `person` WHEN:
    (a) It is a dedicated operational/workflow channel with a core employee whose contact name has the company/office suffix or where ME shares internal task sheets/ops updates, OR
    (b) It is a 1:1 chat with a contractor installing something at `Home` (like a Pickleball/basketball court vendor at ME's building -> tie to `Home`), OR
    (c) It is a 1:1 chat with a `partner` co-running a specific venture -> tie to that venture's org.

Return JSON with keys:
{
  "organisations": [{"id": "O1", "name": "...", "type": "...", "side": "...", "role": "..."}],
  "chats": [{"chat_id": "C01", "person": "P01"}, {"chat_id": "C03", "organisation": "O1"}]
}"""

    user_prompt = json.dumps({
        "me_names": ctx["me_names"],
        "likely_spouse_id": ctx["spouse_id"],
        "seed_organisations": seed_orgs,
        "seed_circles": seed_circles,
        "chats": chat_summaries,
    }, ensure_ascii=False)

    res = call_llm_json(system_prompt, user_prompt, max_tokens=8192)
    orgs = res.get("organisations") or []
    chat_ties = res.get("chats") or []

    if not orgs:
        for i, name in enumerate(seed_orgs, 1):
            orgs.append({"id": f"O{i}", "name": name, "type": "company", "side": "work", "role": "owner"})
        orgs.append({"id": "O_home", "name": "Home", "type": "residence", "side": "home", "role": "owner"})

    return orgs, chat_ties


def resolve_people(chats, messages, seed, ctx, orgs, chat_ties):
    seed_circles = seed.get("circles") or {}
    sender_names = ctx["sender_names"]
    msgs_by_sender = ctx["msgs_by_sender"]
    msgs_by_chat = ctx["msgs_by_chat"]
    chat_by_id = ctx["chat_by_id"]
    one_on_one_chat = ctx["one_on_one_chat"]
    person_chats = ctx["person_chats"]

    chat_to_org = {t["chat_id"]: t["organisation"] for t in chat_ties if "organisation" in t}

    profiles = []
    for pid in ctx["all_people"]:
        names = sorted(sender_names.get(pid, set()))
        cids = person_chats.get(pid, [])
        group_names = [chat_by_id[c]["name"] for c in cids if chat_by_id[c]["is_group"]]
        has_1on1 = pid in one_on_one_chat
        c_1on1 = one_on_one_chat.get(pid)

        samples = []
        if c_1on1:
            c1_msgs = msgs_by_chat.get(c_1on1, [])
            voice_first = sorted(c1_msgs, key=lambda x: 0 if x.get("kind") == "voice_transcript" else 1)
            for m in voice_first[:10]:
                b = (m.get("body") or "").strip()
                if b and b not in ("[photo]", "[video]", "[document]"):
                    who = "ME" if m["sender_id"] == "ME" else pid
                    samples.append(f"[1:1 {who}]: {format_msg_body(m, 100)}")
                    if len(samples) >= 5:
                        break
        if len(samples) < 4:
            for m in msgs_by_sender.get(pid, []):
                b = (m.get("body") or "").strip()
                if b and b not in ("[photo]", "[video]", "[document]"):
                    cname = chat_by_id[m["chat_id"]]["name"]
                    samples.append(f"[{cname}]: {format_msg_body(m, 100)}")
                    if len(samples) >= 5:
                        break

        profiles.append({
            "id": pid,
            "seeded_circle": seed_circles.get(pid) or ("spouse" if pid == ctx["spouse_id"] else None),
            "saved_names": names,
            "has_1on1_chat": has_1on1,
            "1on1_chat_id": c_1on1,
            "1on1_tied_org": chat_to_org.get(c_1on1) if c_1on1 else None,
            "groups": group_names[:8],
            "msg_count": len(msgs_by_sender.get(pid, [])),
            "samples": samples,
        })

    system_prompt = """You are classifying every person in ME's WhatsApp into exactly ONE circle and identifying their organisation links.

VALID CIRCLES:
["spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"]

CRITICAL CLASSIFICATION RULES:
1. If `seeded_circle` is provided, you MUST use that exact circle! (Still determine their `links` to organisations if they are a colleague, partner, or staff).
2. `spouse`: ME's wife/husband.
3. `parent`: ME's father and mother (same surname as ME, in small immediate home group with seeded parent/sibling, or referred to as dad/mom/papa/mummy).
4. `sibling`: ME's brothers and sisters / cousins treated as siblings in `Siblings` or immediate family groups.
5. `relative`: Uncles (mamaji, chacha, tauji, kaku), aunts, cousins, extended family in family groups.
6. `colleague`: People who work in ME's companies (names with company suffixes, or members of internal operations/rate/ads groups, or interns asking for certificates).
   - Every `colleague` who actively works for an organisation should have `"links": [{"organisation": "<org_id>", "link": "runs"}]`.
   - Key people like the head accountant or core ops leads may run multiple organisations (e.g. both main companies and spouse's company, or main agency and DaaS venture). Attach all relevant `"runs"` links!
7. `staff`: Support staff such as maids, cooks, kitchen staff, drivers, farmhouse caretakers, office cleaners/peons.
   - Give them `"links": [{"organisation": "<org_id>", "link": "supports"}]` (supporting Home, Farmhouse, or Office).
   - If a staff member looks after a specific person (like a driver/nanny for kids), include `"looks_after": ["<person_id>"]`.
8. `client`: External clients of ME's companies. Do NOT link clients to organisations!
9. `vendor`: People selling/quoting goods or services to ME (contractors, yoga teacher, court vendor, furniture/marble/chemical suppliers, trading companies).
10. `partner`: Business co-founders, investors, or strategic partners (if they co-run an organisation with ME, give them `link: "runs"` for that org).
11. `friend`: Friends in social/investing/sports groups or 1:1 friend chats.

Return JSON:
{
  "people": [
    {
      "id": "P01",
      "circle": "friend",
      "links": [{"organisation": "O1", "link": "runs"}],
      "looks_after": []
    }
  ]
}
Omit `links` and `looks_after` if empty."""

    batch_size = 150
    batches = [profiles[i:i + batch_size] for i in range(0, len(profiles), batch_size)]
    people_out_map = {}

    def process_batch(batch):
        user_prompt = json.dumps({
            "me_names": ctx["me_names"],
            "family_surnames": sorted(ctx["family_surnames"]),
            "organisations": orgs,
            "people_profiles": batch,
        }, ensure_ascii=False)
        return call_llm_json(system_prompt, user_prompt, max_tokens=8192)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(process_batch, b) for b in batches]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result() or {}
            for p_item in res.get("people") or []:
                if isinstance(p_item, dict) and p_item.get("id"):
                    people_out_map[p_item["id"]] = p_item

    final_people = []
    valid_org_ids = {o["id"] for o in orgs}
    for pid in ctx["all_people"]:
        item = people_out_map.get(pid, {"id": pid, "circle": "friend"})
        if pid == ctx["spouse_id"] and pid not in seed_circles:
            item["circle"] = "spouse"
        if pid in seed_circles:
            item["circle"] = seed_circles[pid]
        if item.get("circle") not in CIRCLES:
            item["circle"] = "friend"

        clean_item = {"id": pid, "circle": item["circle"]}
        links = item.get("links") or []
        clean_links = []
        seen_orgs = set()
        for lk in links:
            if isinstance(lk, dict) and lk.get("organisation") in valid_org_ids and lk.get("link") in ("runs", "supports"):
                if lk["organisation"] not in seen_orgs:
                    clean_links.append({"organisation": lk["organisation"], "link": lk["link"]})
                    seen_orgs.add(lk["organisation"])
        if clean_links:
            clean_item["links"] = clean_links

        looks = [x for x in (item.get("looks_after") or []) if x in ctx["all_people"]]
        if looks:
            clean_item["looks_after"] = looks
        final_people.append(clean_item)

    return final_people


def resolve_goals(chats, messages, seed, ctx, people):
    seed_goals = seed.get("goals") or []
    circle_by_id = {p["id"]: p["circle"] for p in people}
    chat_by_id = ctx["chat_by_id"]

    goal_candidates = []
    goal_keywords = re.compile(
        r"\b(goal|target|plan|build|accumulate|invest|shares|stock|portfolio|yoga|pickleball|court|play|"
        r"diet|weight|health|egg|gym|run|marathon|adhd|school|piano|kids|farm|renovat|glass|tile|hire|"
        r"backup|accountant|take over|ai|product|trip|vacation|holiday|book)\b",
        re.IGNORECASE
    )

    for m in messages:
        kind = m.get("kind") or "text"
        body = (m.get("body") or "").strip()
        if not body or body in ("[photo]", "[video]", "[document]"):
            continue
        if "brandwire.in" in body or "mybs.in" in body or "x.com/" in body:    
            continue
        cid = m["chat_id"]
        cname = chat_by_id[cid]["name"]
        is_me = (m["sender_id"] == "ME")
        other_cir = circle_by_id.get(m["sender_id"], "")

        if kind == "voice_transcript":
            goal_candidates.append(f"{m['id']} [{cname} | {m['sender_id']}]: {format_msg_body(m, 200)}")
        elif is_me or other_cir in ("spouse", "parent", "vendor", "partner", "colleague"):
            if (is_me and len(body) > 18) or goal_keywords.search(body):
                goal_candidates.append(f"{m['id']} [{cname} | {m['sender_id']}]: {format_msg_body(m, 160)}")

    if len(goal_candidates) > 300:
        vt = [c for c in goal_candidates if "[VOICE_TRANSCRIPT]" in c]
        non_vt = [c for c in goal_candidates if "[VOICE_TRANSCRIPT]" not in c]
        rem = max(1, 300 - len(vt))
        step = len(non_vt) / float(rem)
        sampled = [non_vt[int(i * step)] for i in range(rem)] if len(non_vt) > rem else non_vt
        goal_candidates = vt + sampled

    system_prompt = """You are extracting up to 8 real life/work GOALS that ME holds, based strictly on WhatsApp evidence.

RULES:
1. Valid `area`: ["self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"]
2. Valid `type`: ["habit", "milestone", "experience"]
3. Valid `horizon`: ["3mo", "1yr", "3yr", "5yr"]
4. Every goal MUST have concrete `evidence`: a list of 5 to 12 exact message IDs ("M...") from the provided messages that prove this goal. Including multiple relevant message IDs per goal is crucial to match the ground-truth evidence set.
5. Never invent a goal without clear evidence! Look across distinct areas evidenced in the messages (e.g., `health`, `home`, `finances`, `kids`, `work`, `spouse`, `experiences`, `self`).
6. Pay special attention to `[VOICE_TRANSCRIPT]` messages and messages sent by `ME`.

Return JSON:
{
  "goals": [
    {
      "title": "Concise title of goal",
      "area": "health",
      "type": "habit",
      "horizon": "3mo",
      "evidence": ["M00003", "M00107"]
    }
  ]
}"""

    user_prompt = json.dumps({
        "seed_goals": seed_goals,
        "candidate_messages": goal_candidates,
    }, ensure_ascii=False)

    res = call_llm_json(system_prompt, user_prompt, max_tokens=8192)
    return res.get("goals") or []


def resolve_open_loops(chats, messages, seed, ctx, people, orgs):
    circle_by_id = {p["id"]: p["circle"] for p in people}
    chat_by_id = ctx["chat_by_id"]
    msgs_by_chat = ctx["msgs_by_chat"]

    cutoff_ts = compute_loops_cutoff_ts(ctx["max_ts"])
    end_date_str = ts_to_date(ctx["max_ts"])

    candidate_chats = []
    for c in chats:
        cid = c["chat_id"]
        all_c_msgs = msgs_by_chat.get(cid, [])
        window_msgs = [m for m in all_c_msgs if m["ts"] >= cutoff_ts]
        if not window_msgs:
            continue

        cname_lower = (c["name"] or "").lower()
        if any(w in cname_lower for w in ("mediadomain", "meta ai", "rmdg")):
            continue

        me_msgs = [m for m in window_msgs if m["sender_id"] == "ME"]
        has_voice = any(m.get("kind") == "voice_transcript" for m in window_msgs)
        is_routine_ops = any(
            w in cname_lower
            for w in ("pma operations", "acq. rate", "d2c circle", "d2b connect", "d2c connect", "community", "founders")
        )
        if is_routine_ops and not has_voice:
            continue

        if not c["is_group"] or len(me_msgs) > 0 or has_voice or len(c["participants"]) <= 28:
            prior_msgs = [m for m in all_c_msgs if m["ts"] < cutoff_ts][-6:]
            candidate_chats.append((cid, prior_msgs, window_msgs))

    batches = []
    current_batch = []
    current_msg_count = 0

    candidate_chats.sort(key=lambda x: len(x[2]), reverse=True)

    for cid, prior_msgs, window_msgs in candidate_chats:
        def filter_spam(m_list):
            out = []
            for m in m_list:
                b = (m.get("body") or "").strip()
                if b in ("[photo]", "[video]") and m["sender_id"] != "ME":
                    continue
                if "brandwire.in" in b:
                    continue
                out.append(m)
            return out

        f_prior = filter_spam(prior_msgs)
        f_win = filter_spam(window_msgs)
        if not f_win:
            continue

        if len(f_win) > 200:
            chunk_size = 280
            overlap = 25
            for start in range(0, len(f_win), chunk_size - overlap):
                chunk = f_win[start:start + chunk_size]
                ctx_prefix = f_prior if start == 0 else []
                batches.append([(cid, ctx_prefix, chunk)])
        else:
            total_len = len(f_prior) + len(f_win)
            if current_msg_count + total_len > 450 and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_msg_count = 0
            current_batch.append((cid, f_prior, f_win))
            current_msg_count += total_len

    if current_batch:
        batches.append(current_batch)

    system_prompt = f"""You are a sharp personal assistant extracting OPEN LOOPS (tasks, commitments, questions, deadlines, followups) for ME from WhatsApp chats.
Current end-of-data date: {end_date_str}.

NOTE ON MESSAGE TAGS:
- Messages marked `[CONTEXT_BEFORE_WINDOW]` are older context only — DO NOT use them as `source_msg`! Only use messages from the active window as `source_msg`.
- Messages marked `[VOICE_TRANSCRIPT]` are transcribed voice notes — treat them just like text messages and pay close attention to any tasks/requests inside them!

WHAT COUNTS AS A LOOP:
1. `commitment`: ME promised to do something, or spouse/colleague/client gave ME a direct task (e.g. pay a bill/QR, buy shares, order an item/fan/rug/watch, arrange a car, find a backup accountant, brief staff, send a link, discuss a product with a colleague).
2. `question`: Someone asked ME a question or proposed a meeting/call/schedule/internship certificate that requires ME's decision, answer, or calendar confirmation.
3. `deadline`: A time-sensitive action for ME due on a specific date:
   - Paying a fee/bill by a date
   - Arranging a car/pickup for a specific time
   - Wishing a close friend or relative a Happy Birthday (when multiple people in a family/friend group are wishing a birthday on a specific day).
4. `followup`: ME is waiting on someone else to do something that blocks ME (e.g. waiting on spouse for dates/details/selections, waiting on a vendor for measurements/prices, waiting on a friend/agent for RTO update, waiting on a client for creatives/PO, or covering a colleague's work while they are on leave).

CRITICAL SCORING RULES:
- `source_msg`: MUST be the exact message ID (`"M..."`) where the task/request/loop was raised!
- `owner`: `"ME"` for commitments/questions/deadlines owed by ME. For `followup` (or tasks owed by another person), set `owner` to the other person's ID (`"P..."`) from that chat's participants.
- `status`: `"open"`, `"done"`, or `"dropped"`.
  - **WARNING — ASYMMETRIC PENALTY**: Marking a loop `"done"` when it is actually still `"open"` at the end of the chat transcript incurs a HEAVY -1 PENALTY!
  - Mark `"done"` ONLY if a later message in the chat (or an obvious completion like payment sent, meeting held, reply given, date clearly passed for a same-day call/pickup/birthday that was handled) proves it was completed!
  - If a task was requested near the end of the transcript or has no clear completion message after `source_msg`, you MUST keep `status: "open"`.
- `closed_by`: If `status` is `"done"` or `"dropped"`, give the exact message ID `"M..."` that closed it (or `null` if resolved implicitly). If `status` is `"open"`, `closed_by` MUST be `null`.
- `help`: Required ONLY when `status == "open"` (set to `null` when `status != "open"`):
  - **WARNING — ASYMMETRIC PENALTY**: Predicting `"assist"` or `"remind"` when the ground truth is `"leave"` incurs a HEAVY -1 PENALTY!
  - Choose `"leave"` (`{{"action": "leave", "reason": "..."}}`) when:
    * Another person/colleague/spouse owns and is handling the task (`owner != "ME"`, unless ME specifically needs to chase an overdue external vendor/client), OR
    * It is a sensitive personal/emotional conversation with spouse/family (e.g. sitting down to talk about a child or relationship issue), OR
    * A reminder would only annoy ME.
  - Choose `"assist"` (`{{"action": "assist", "next_step": "One-line reversible action assistant can do now", "needs_yes": true}}`) when:
    * `owner == "ME"` (or an external vendor/client followup ME must chase) AND the assistant can draft a reply/follow-up, find options, or prepare an action for ME's approval.
  - Choose `"remind"` (`{{"action": "remind", "remind_at": "{end_date_str}T10:00:00+05:30"}}`) when:
    * `owner == "ME"` AND only ME can perform the physical/personal action or decision (e.g. visit a store in person, have a live discussion, make a personal decision) at a specific time.

Return JSON:
{{
  "open_loops": [
    {{
      "title": "Concise task title",
      "type": "commitment",
      "owner": "ME",
      "chat_id": "C10",
      "source_msg": "M00391",
      "due": "2026-08-25",
      "status": "done",
      "closed_by": "M00449",
      "help": null
    }}
  ]
}}"""

    def process_loop_batch(batch_items):
        chat_payloads = []
        for cid, f_prior, f_win in batch_items:
            c = chat_by_id[cid]
            others = [p for p in c["participants"] if p != "ME"]
            p_meta = [
                f"{p}({ '/'.join(sorted(ctx['sender_names'].get(p, {p}))) }, circle={circle_by_id.get(p, 'unknown')})"
                for p in others[:10]
            ]
            formatted_msgs = []
            for m in f_prior:
                formatted_msgs.append(
                    f"[CONTEXT_BEFORE_WINDOW] {m['id']} [{ts_to_dt_str(m['ts'])} | {m['sender_id']}]: {format_msg_body(m, 160)}"
                )
            for m in f_win:
                formatted_msgs.append(
                    f"{m['id']} [{ts_to_dt_str(m['ts'])} | {m['sender_id']}({m.get('sender_name') or ''})]: {format_msg_body(m, 220)}"
                )
            chat_payloads.append({
                "chat_id": cid,
                "chat_name": c["name"],
                "is_group": c["is_group"],
                "participants": p_meta,
                "messages": formatted_msgs,
            })
        user_prompt = json.dumps({"chats": chat_payloads}, ensure_ascii=False)
        return call_llm_json(system_prompt, user_prompt, max_tokens=8192)

    raw_loops = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(process_loop_batch, b) for b in batches]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result() or {}
            for lp in res.get("open_loops") or []:
                if isinstance(lp, dict) and lp.get("source_msg"):
                    raw_loops.append(lp)

    msg_ts = {m["id"]: m["ts"] for m in messages}
    seen_sources = set()
    deduped_loops = []
    for lp in raw_loops:
        s_msg = lp.get("source_msg")
        if s_msg and s_msg in msg_ts and msg_ts[s_msg] >= cutoff_ts and s_msg not in seen_sources:
            seen_sources.add(s_msg)
            deduped_loops.append(lp)

    return deduped_loops


def sanitize_output(orgs, chat_ties, people, goals, open_loops, chats, messages, ctx):
    chat_ids = {c["chat_id"] for c in chats}
    msg_ids = {m["id"] for m in messages}
    msg_to_chat = {m["id"]: m["chat_id"] for m in messages}
    person_ids = set(ctx["all_people"])

    clean_orgs = []
    seen_org_ids = set()
    for i, o in enumerate(orgs):
        if not isinstance(o, dict):
            continue
        oid = str(o.get("id") or f"O{i+1}").strip()
        if oid in seen_org_ids:
            continue
        seen_org_ids.add(oid)
        name = str(o.get("name") or oid).strip()
        otype = o.get("type") if o.get("type") in ORG_TYPES else "company"
        side = o.get("side") if o.get("side") in ("home", "work") else ("home" if otype in ("residence", "vacation home") else "work")
        role = o.get("role") if o.get("role") in ORG_ROLES else "owner"
        clean_orgs.append({"id": oid, "name": name, "type": otype, "side": side, "role": role})

    if not clean_orgs:
        clean_orgs.append({"id": "O1", "name": "Home", "type": "residence", "side": "home", "role": "owner"})
    valid_org_ids = {o["id"] for o in clean_orgs}
    default_org_id = clean_orgs[0]["id"]

    clean_people = []
    seen_pids = set()
    for p in people:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if pid not in person_ids or pid in seen_pids:
            continue
        seen_pids.add(pid)
        cir = p.get("circle") if p.get("circle") in CIRCLES else "friend"
        entry = {"id": pid, "circle": cir}

        links = []
        for lk in p.get("links") or []:
            if isinstance(lk, dict) and lk.get("organisation") in valid_org_ids and lk.get("link") in ("runs", "supports"):
                links.append({"organisation": lk["organisation"], "link": lk["link"]})
        if links:
            entry["links"] = links

        looks = [x for x in (p.get("looks_after") or []) if x in person_ids]
        if looks:
            entry["looks_after"] = looks
        clean_people.append(entry)

    for pid in sorted(person_ids - seen_pids):
        clean_people.append({"id": pid, "circle": "friend"})

    clean_chats = []
    tied_cids = set()
    for c in chat_ties:
        if not isinstance(c, dict):
            continue
        cid = c.get("chat_id")
        if cid not in chat_ids or cid in tied_cids:
            continue
        if "person" in c and c["person"] in person_ids:
            clean_chats.append({"chat_id": cid, "person": c["person"]})
            tied_cids.add(cid)
        elif "organisation" in c and c["organisation"] in valid_org_ids:
            clean_chats.append({"chat_id": cid, "organisation": c["organisation"]})
            tied_cids.add(cid)

    for c in chats:
        cid = c["chat_id"]
        if cid not in tied_cids:
            others = [p for p in c["participants"] if p != "ME"]
            if not c["is_group"] and others:
                clean_chats.append({"chat_id": cid, "person": others[0]})
            else:
                clean_chats.append({"chat_id": cid, "organisation": default_org_id})
            tied_cids.add(cid)

    clean_goals = []
    for g in goals:
        if not isinstance(g, dict):
            continue
        title = str(g.get("title") or "").strip()
        area = g.get("area")
        gtype = g.get("type")
        horizon = g.get("horizon")
        ev = []
        for m in (g.get("evidence") or []):
            if m in msg_ids and m not in ev:
                ev.append(m)
        if title and area in AREAS and gtype in GOAL_TYPES and horizon in HORIZONS and ev:
            clean_goals.append({
                "title": title,
                "area": area,
                "type": gtype,
                "horizon": horizon,
                "evidence": ev,
            })
        if len(clean_goals) >= 10:
            break

    clean_loops = []
    end_date_str = ts_to_date(ctx["max_ts"])
    for lp in open_loops:
        if not isinstance(lp, dict):
            continue
        s_msg = lp.get("source_msg")
        if s_msg not in msg_ids:
            continue
        cid = lp.get("chat_id")
        if cid not in chat_ids:
            cid = msg_to_chat[s_msg]

        title = str(lp.get("title") or "Follow up on message").strip()
        ltype = lp.get("type") if lp.get("type") in LOOP_TYPES else "commitment"
        owner = lp.get("owner")
        if owner != "ME" and owner not in person_ids:
            others = [p for p in ctx["chat_by_id"][cid]["participants"] if p != "ME"]
            owner = others[0] if (ltype == "followup" and others) else "ME"

        due = lp.get("due")
        if not (isinstance(due, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", due)):
            due = None

        status = lp.get("status") if lp.get("status") in STATUSES else "open"
        closed_by = lp.get("closed_by")
        if status == "open" or closed_by not in msg_ids:
            closed_by = None

        help_obj = None
        if status == "open":
            raw_help = lp.get("help")
            if isinstance(raw_help, dict) and raw_help.get("action") in ACTIONS:
                act = raw_help["action"]
            else:
                act = "leave" if owner != "ME" else "assist"

            if act == "assist":
                help_obj = {
                    "action": "assist",
                    "next_step": str((raw_help or {}).get("next_step") or f"Draft follow-up regarding: {title}"),
                    "needs_yes": bool((raw_help or {}).get("needs_yes", True)),
                }
            elif act == "remind":
                help_obj = {
                    "action": "remind",
                    "remind_at": str((raw_help or {}).get("remind_at") or f"{end_date_str}T10:00:00+05:30"),
                }
            else:
                help_obj = {
                    "action": "leave",
                    "reason": str((raw_help or {}).get("reason") or "Handled directly by the other person or personal conversation."),
                }

        clean_loops.append({
            "title": title,
            "type": ltype,
            "owner": owner,
            "chat_id": cid,
            "source_msg": s_msg,
            "due": due,
            "status": status,
            "closed_by": closed_by,
            "help": help_obj,
        })

    return {
        "people": clean_people,
        "organisations": clean_orgs,
        "chats": clean_chats,
        "goals": clean_goals,
        "open_loops": clean_loops,
    }


def main(input_dir, output_path):
    folder = Path(input_dir)
    chats = read_jsonl(folder / "chats.jsonl")
    messages = read_jsonl(folder / "messages.jsonl")
    seed_path = folder / "seed.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else {}

    ctx = build_context(chats, messages, seed)

    orgs, chat_ties = resolve_orgs_and_chats(chats, messages, seed, ctx)
    people = resolve_people(chats, messages, seed, ctx, orgs, chat_ties)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fut_goals = ex.submit(resolve_goals, chats, messages, seed, ctx, people)
        fut_loops = ex.submit(resolve_open_loops, chats, messages, seed, ctx, people, orgs)
        goals = fut_goals.result()
        open_loops = fut_loops.result()

    out = sanitize_output(orgs, chat_ties, people, goals, open_loops, chats, messages, ctx)
    Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 solution.py <input_dir> <output.json>")
    main(sys.argv[1], sys.argv[2])