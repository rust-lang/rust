# Execute one Beads task

## Your role

You are a coding agent. For this invocation you read **one** task identified by
`TASK_ID` and carry out the work described there. Nothing else: no PR workflow,
no iteration caps or retry “loops,” no merge or review-comment handling unless
the task text itself asks for it.

You do **one** task per invocation, then stop and report.

---

## Input

- `TASK_ID` (required) — the Beads task id to execute

If `TASK_ID` is missing or empty, stop and report:
`Missing required parameter: TASK_ID`.

---

## Prerequisites

- `TASK_ID` is provided
- `bd` is available if you use it to load the task (optional but typical)

---

## Workflow

### 1. Load the task

```bash
bd show <task-id> --json
```

Use the given `TASK_ID` as `<task-id>`. Do not pick a different task.

If the task does not exist, report that and stop.

Read the title, description, acceptance criteria, dependencies, and recent
comments. If something is ambiguous, add a short `bd comment` and stop rather
than guessing.

### 2. Claim (optional but recommended)

```bash
bd update <task-id> --claim
```

If claim fails, note it and either stop or continue only if the task is clearly
yours to do per team practice.

### 3. Implement

- Inspect the relevant code before changing it.
- Stay within the task’s scope.
- Run checks/tests appropriate to the change (project defaults, e.g. `./x.py
  check` and targeted tests when they match the task).
- Commit with clear messages when you have coherent units of work.

Do **not** open or update pull requests as part of this prompt. Do **not** run a
fixed “try N times then give up” loop; fix issues until the work matches the
task or you hit a genuine blocker you report.

### 4. Report

Summarize briefly:

- Task id and title
- What you did
- How you verified (commands, outcome)
- Blockers, if any (with enough detail to continue later)

---

## Hard rules

- Use only the provided `TASK_ID`; do not auto-select from `bd ready` or lists.
- One task per invocation; do not chain the next task automatically.
- Do not force-push to protected/default branches unless the task explicitly
  requires it and you have agreement to do so.

---

## Exit tokens (optional)

- When the invocation finishes successfully for the task as described, you may
  print: `TASK_SUCCESS`
- On failure, missing task, or abort: `TASK_FAILED`
