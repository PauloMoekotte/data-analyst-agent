# data-analyst

Conversational data-analyst agent, built with the Mistral Agents SDK and
deployed on Mistral Cloud as a managed deployment.

You give it a link to a CSV file in your Google Drive and then ask questions in
plain language. It fetches the file through your connected Google Drive
connector, loads it, and answers with real numbers from the data.

## How it works

The agent runs with **no sandbox and no code execution**. It has exactly four
fixed pandas-backed tools, plus the `google_drive_mcp` connector for Drive:

| Tool | Purpose |
| --- | --- |
| `load_csv` | Load the CSV (base64 or raw text) as the session's dataset |
| `describe_columns` | Per-column type, nulls, unique/min/max/mean, top values |
| `filter_rows` | Row filters (AND-ed conditions: `==`, `>=`, `in`, `contains`, `is_null`, …) with a sample of matched rows |
| `groupby_aggregate` | Group by with count/sum/mean/median/min/max/std/nunique, optional filters, sorting and limits |

Dataset state is kept in memory per session id (`ToolCallbackContext.session.id`),
so follow-up questions reuse the loaded file without re-uploading. If the worker
restarts and the cache is gone, the tools say so clearly and the agent reloads
the file automatically. Session history itself (everything discussed) is kept by
the platform: deployed sessions are durable and end at most 24 hours after
their last activity.

Downloading the file works as: `download_file_content` (Drive connector) →
`csv_base64` into `load_csv`. The model chains the two calls, so the CSV bytes
never get typed out by the model twice.

## Local development

```bash
uv run --env-file .env agents chat
```

## Deploy (Mistral-managed)

`agents deploy` registers the agent as a Workflows managed deployment. It has
connectors, so the deployment runs hardened by default.

Requirements:

- the project lives in a GitHub repository and the **GitHub connector** is
  connected on your account in Mistral Studio;
- a `MISTRAL_API_KEY` whose user has a workspace role with Workflows access
  (e.g. Developer) — put it in `.env` (gitignored).

```bash
uv run --env-file .env agents deploy
```

After the deploy:

- chat in Vibe Code Web at the URL the deploy prints, of the form
  `https://chat.mistral.ai/managed-agents/data-analyst`
- watch sessions and traces in Mistral Studio → Managed Agents.

Use agents by name from any Python app:

```python
agent = agents.Agent.load_from(agent_name="data-analyst")
session = await agent.start_session()
```

## Update

Commit and push, then run `uv run --env-file .env agents deploy` again.
