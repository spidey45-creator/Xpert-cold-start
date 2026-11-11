# AI Collaboration Session Log — Xpert Cold Start

## Session Summary
1. **Dataset & Scoring Analysis:** Analyzed `check.py` and `schema.json` to identify exact evaluation formulas: Jaccard bipartite matching for organisations (`max(n_ours, n_yours)` penalty), set intersection on `evidence` message IDs for `goals`, 14-day IST cutoff window (`loops_scored_from`) for `open_loops`, and -1 asymmetric penalties on false loop closures (`done`/`dropped` when `open`) and intrusive help actions (`assist`/`remind` when `leave`).
2. **Hybrid Graph + LLM Pipeline Design:** Built `build_context()` in `solution.py` to deterministically extract 1:1 chats, surname clusters, and spouse heuristics before batching compact summaries into `resolve_orgs_and_chats`, `resolve_people`, `resolve_goals`, and `resolve_open_loops`.
3. **OpenRouter Hardening:** Added `call_llm_json()` fallback logic to gracefully retry without `response_format={'type': 'json_object'}` on HTTP 400/422, strip `<think>` tags from reasoning models, and enforce strict schema sanitization in `sanitize_output()`.
