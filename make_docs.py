#!/usr/bin/env python3
"""Writes WRITEUP.md (from your real usage.jsonl + check.py score) and .gitignore.
Run AFTER a full dev run:   python make_docs.py <model-id> <input_price_per_1M> <output_price_per_1M>
Example:                    python make_docs.py google/gemini-2.5-flash 0.30 2.50   (check prices on the model's OpenRouter page)
"""
import json, subprocess, sys
from pathlib import Path

if len(sys.argv) != 4: sys.exit(__doc__)
model, pin, pout = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

calls = pt = ct = 0; secs = 0.0
if Path("usage.jsonl").exists():
    for l in open("usage.jsonl", encoding="utf-8"):
        if l.strip():
            u = json.loads(l); calls += 1; pt += u.get("prompt_tokens") or 0; ct += u.get("completion_tokens") or 0; secs += u.get("seconds") or 0
nmsg = sum(1 for _ in open("data/dev/messages.jsonl", encoding="utf-8")) if Path("data/dev/messages.jsonl").exists() else 0
cost = pt / 1e6 * pin + ct / 1e6 * pout
per1k = cost / nmsg * 1000 if nmsg else 0

score = "(run check.py first)"
for out in ("output.json", "out.json"):
    if Path(out).exists():
        r = subprocess.run([sys.executable, "check.py", "data/dev", out], capture_output=True, text=True)
        score = "\n".join(l for l in r.stdout.splitlines() if "/" in l and "Cost" not in l)
        break

W = f"""# Write-up

**Model (OpenRouter id):** `{model}`

## What I focused on, and why
Rules first, then a model only where rules cannot decide. Points are spread over circles, organisations,
chats, loops and help, and the checker punishes some mistakes more than others, so I built around that:
a false "done" costs a point (so a loop is only closed when a real later message in the same chat shows it),
and `assist`/`remind` on something Arjun wants left alone costs a point (so anything owned by someone else
defaults to `leave`). Extra organisations lower the organisation score, so empty ones are dropped.
Goals are capped at 6 and each needs real message evidence.

## Approach
1. `rules_v1.py` (no model): ties 1:1 chats to the other person, builds Home and the seeded organisations by
   name matching, guesses circles from saved names and group names. This is also the fallback, so the program
   always writes a valid file even if the model fails.
2. Organisations: one call clusters group chats (name, size, top senders) into organisations.
3. Circles: calls of 6 people each, with a compact profile (chats, sample messages). Seeded circles always win.
4. Goals: one call over Arjun's own goal-like messages.
5. Open loops and help: one call per chat that has candidate messages. Every answer is validated: message ids
   must exist in that chat, closings need a later closing message, owners and dates are checked, enums repaired.
All calls are cached on disk, retried with backoff, and stop after 24 minutes.

## What it costs
Measured on the dev set ({nmsg} messages): {calls} calls, {pt:,} prompt tokens, {ct:,} completion tokens,
about {secs/60:.1f} minutes of call time (sequential sum). At ${pin}/1M input and ${pout}/1M output that is
about ${cost:.3f} in total, **${per1k:.4f} per 1,000 messages**. Almost all tokens go to the per-chat loop calls.

Dev score (parts the checker can score):
```
{score}
```

## Where it's wrong
These are expectations, not measurements; I will compare them with the hidden set.
- Loops: the model will miss some and invent some; I expect precision and recall well below 1.
- "ok"/"done" replies are ambiguous, so some loops stay open that are really done (cheap) and a few get closed wrongly (costly).
- Circles: vendor vs client and relative vs friend will be confused for people with little text.
- Organisations: quiet ones (an office or holiday home with few chats) are hard to find from chats alone.
- Goals: few weeks of chat shows few goals; I may miss real ones or over-read plans as goals.
"""
Path("WRITEUP.md").write_text(W, encoding="utf-8")
Path(".gitignore").write_text("data/\n*.zip\nusage.jsonl\noutput.json\nout.json\n.env\n.llm_cache/\n__pycache__/\n*.pyc\n", encoding="utf-8")
Path("ai-logs").mkdir(exist_ok=True)
print("Wrote WRITEUP.md and .gitignore. Now put your REAL AI chat exports in ai-logs/ (not written by me or any script).")