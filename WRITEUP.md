# Xpert Cold Start — Technical Write-Up

**Selected OpenRouter Model:** `google/gemini-2.5-flash`

## 1. What We Focused On & Why (Scoring & Asymmetric Penalty Alignment)
Our architecture is directly engineered around the exact scoring mechanics and asymmetric penalties in `check.py`:
- **Bounded Organisation Discovery (8 pts):** `match_orgs()` performs greedy 1-to-1 bipartite matching via Jaccard similarity over attached chats and people, divided by `max(n_ours, n_yours)`. Over-predicting organisations for client companies or casual social groups inflates `n_yours` and directly penalizes the score. We strictly constrain discovery to ME's 5-8 core work/home entities (seeded companies, co-run sub-ventures, spouse's business where ME helps, `Office`, `Home`, and `Farmhouse`).
- **Conservative Loop Closure Policy (14 pts):** Marking an open loop as `done` or `dropped` when ground truth is `open` incurs a harsh **-1 penalty**, whereas leaving a completed loop as `open` scores `0`. Our prompt and post-processing enforce a strict proof-of-completion threshold: unless an explicit closing message (`closed_by`) exists after `source_msg`, the loop defaults to `status: "open"`.
- **Risk-Averse Help Policy (10 pts):** Predicting `assist` or `remind` when ground truth is `leave` incurs a **-1 penalty**. Whenever `owner != "ME"` (routine colleague/spouse tasks) or the topic involves sensitive personal/family matters, our pipeline defaults to `action: "leave"`.
- **High-Recall Goal Evidence (8 pts):** Goal matching requires matching `area` and a non-empty intersection (`set(g["evidence"]) & set(a["evidence"])`). We supply 5-12 high-signal message IDs per goal (prioritizing `ME`'s messages and `voice_transcript` notes) to maximize set intersection without penalty.

## 2. System Architecture (`solution.py`)
The pipeline is a hybrid **Deterministic Graph Pre-processor + Multi-Stage Batched LLM Engine** using only the Python 3.9+ standard library:
1. **Deterministic Graph & Context Extraction (`build_context`):** Parses `chats.jsonl` and `messages.jsonl` to build 1:1 chat mappings, participant co-occurrence graphs, sender alias sets, and shared family surname clusters. Uses bidirectional message volume, surname overlap, and shared family group membership to deterministically identify `spouse` (`P13` in dev) when unseeded.
2. **Joint Organisation & Chat Resolution (`resolve_orgs_and_chats`):** Compresses all chats into token-efficient metadata profiles (participant previews, seeded circle tags, message counts, and top 4 non-trivial messages) and resolves all organisations and `chats` ties in a single global-context LLM call.
3. **Batched People & Link Classification (`resolve_people`):** Groups all non-ME participants into batches with 1:1 chat snippets (prioritizing `voice_transcript` messages), group memberships, and tied organisation IDs. Enforces seeded circles and attaches `runs` (for colleagues/partners) and `supports` (for household/office/farmhouse staff) links.
4. **Temporal-Windowed Open Loop & Goal Extraction (`resolve_open_loops`, `resolve_goals`):**
   - Computes `cutoff_ts` (14-day scoring window prior to `max_ts` in IST) and includes up to 6 prior messages tagged `[CONTEXT_BEFORE_WINDOW]` so the LLM understands context without emitting out-of-window `source_msg` IDs.
   - Filters out noisy broadcast/routine ops channels unless `voice_transcript` messages are present.
5. **Schema Sanitizer & Fallback Layer (`sanitize_output`):** Guarantees 100% `schema.json` compliance, strips `<think>` blocks or markdown fences, retries with `response_format=None` if an OpenRouter provider rejects JSON mode, deduplicates `source_msg` IDs, and ensures every chat and person is valid.

## 3. Cost & Runtime Breakdown (`google/gemini-2.5-flash`)
- **Total LLM Calls on Dev (~4,500 messages):** ~11 batched calls (`temperature=0`).
- **Token Usage on Dev:** ~42,000 prompt tokens + ~9,500 completion tokens.
- **Estimated Cost:** ~$0.013 total on `dev` (**~$0.0029 per 1,000 messages** at Gemini 2.5 Flash rates of $0.15/1M input, $0.60/1M output), finishing in **~45-75 seconds** (well under the 30-minute limit).

## 4. Predicted Failure Modes & Expected Frequencies on Hidden Set
1. **Off-Channel / Implicit Loop Closures (~15-20% of open loops):** Commitments or questions resolved via phone call, in-person conversation, or in a separate 1:1 thread leave no textual confirmation in the originating chat. Because of the -1 false-closure penalty, our system intentionally predicts `status: "open"` on these cases.
2. **1:1 Operational Chat vs. Person Tie Ambiguity (~5-8% of chats):** Distinguishing whether a 1:1 chat with a senior employee, partner, or home vendor (e.g., court installation contractor) should be tied to the `organisation` or the `person` when operational and personal messages mix.
3. **Low-Signal Passive Group Members (~8-12% of unseeded people):** Contacts who appear only in mixed social/family/community groups with 0-1 short messages (`"Ok"`, `"Happy birthday"`) lack lexical signal to reliably separate `friend`, `relative`, and `client`.
