# Cold Start: starter kit

The challenge is on https://goals.xpert.chat/w/intern. Read it first. This kit gets you running
before the real data arrives.

## What's here

| File | What it is |
|---|---|
| `run.sh` | The one command we run: `./run.sh <input_dir> <output.json>`. It must finish in 30 minutes |
| `solution.py` | A naive starting point that writes a valid output. It scores badly on purpose. Replace it |
| `llm.py` | A tiny client for any OpenAI-compatible API. It logs token use to `usage.jsonl` |
| `schema.json` | The exact shape of `output.json` |
| `check.py` | Checks your output's shape, and scores it if the folder has answers |
| `example/` | A tiny made-up person, Meera: 6 chats, 29 messages, with the answers |
| `WRITEUP.md` | The template for your write-up |
| `ai-logs/` | Where your AI session transcripts go |

Python 3.9 or later. Nothing to install. You can write your solution in any language, as long as
`run.sh` runs it on a clean Linux machine.

## Try it now

```bash
./run.sh example out.json
python3 check.py example out.json
```

The checker prints the score for each part. The naive solution gets the chats right and little
else. That's your starting line.

## When the real data arrives

Download it from the challenge page and unzip it inside this folder. It makes `data/dev/` (already in `.gitignore`):

```bash
./run.sh data/dev out.json
python3 check.py data/dev out.json
```

`data/dev` is the first two weeks with their answers. We run your code on the six weeks after, which
you never see, with every name changed again. Open loops are scored on its last two weeks.

## The input

```
chats.jsonl      chat_id, name, is_group, participants (person ids; Arjun is "ME")
messages.jsonl   id, chat_id, sender_id, sender_name, ts (unix seconds), kind, body, quoted_id
seed.json        what the person told us at signup: some circles, some organisation names, a goal
answers/         dev set only: answers.json
```

`kind` is `text`, `voice_transcript`, `image_caption` or `document`. `quoted_id` is always null: the export kept no quoted replies. Voice transcripts are rare: none
in the dev set, about 50 in the hidden set.

## The output

See `schema.json`, and the challenge page for what each part means. In short:

```json
{
  "people":        [{"id": "P03", "circle": "staff", "links": [{"organisation": "home", "link": "supports"}], "looks_after": ["P04"]}],
  "organisations": [{"id": "home", "name": "Home", "type": "residence", "side": "home", "role": "owner"}],
  "chats":         [{"chat_id": "C03", "organisation": "home"}, {"chat_id": "C01", "person": "P01"}],
  "goals":         [{"title": "Run the half marathon in January", "area": "health", "type": "milestone", "horizon": "3mo", "evidence": ["M026"]}],
  "open_loops":    [{"title": "Decide whether to take Zest", "type": "question", "owner": "ME", "chat_id": "C04",
                     "source_msg": "M016", "due": "2026-10-26", "status": "open", "closed_by": null,
                     "help": {"action": "remind", "remind_at": "2026-10-25T10:00:00+05:30"}}]
}
```

## How the checker scores

The same rules we use on the hidden set. They cover 74 of the 100 points.

- **Circles (12):** share of people with the right circle. People whose circle is in `seed.json`
  don't count.
- **Organisations (8):** we pair your organisations with ours by overlap of the chats and people
  attached to each. Extra or missing organisations lower the score.
- **Chats (8):** share of chats tied to the right person, or to your organisation paired with ours.
- **Goals (8):** a goal matches ours when the area is the same and at least one evidence message
  is shared. The score balances how many of yours are right with how many of ours you found. The
  goal in `seed.json` doesn't count.
- **Open loops found (14):** a loop matches ours when your `source_msg` is one of the messages we
  accept for it. The score balances how many of yours are real with how many of ours you found.
- **Open loops closed right (14):** for each loop you found, the right status counts 1. Calling a
  loop done or dropped when it's still open costs 1. Divided by all of our loops.
- **Where to help (10):** for each open loop we labelled, the right action counts 1. Saying
  `assist` or `remind` where we said `leave` costs 1.

Cost (12), the write-up (7) and runs with one command (7) are scored by us.

## The answers format

`answers/answers.json` has the same shape as your output, with three differences in `open_loops`
(and, on the hidden set, a `loops_scored_from` time: loops raised before it are not scored):
`source_msgs` lists every message we accept as the source, `closing_msgs` lists the messages that
show it closed, and `help` is just the action, with a `why`.

## Cost

The hidden set is about 15,000 messages, so plan for that size: batch your calls.

Use any model with an OpenAI-compatible API. Read `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`
from the environment. When we score, we set them to our own key and the model you named, and count
every token. `llm.py` does this for you and logs what each call used.
