# Concept Mapping Review Guidelines

> **Status**: Living reference — Phase 9 baseline.
> Derived from `docs/migration/concept-mapping.md`.
> Use this checklist when reviewing any PR that touches `kernel/`, `abi/`,
> `bran/`, `stem/`, or `userspace/`.

---

## Purpose

This document translates the naming rules from `docs/migration/concept-mapping.md`
into an actionable PR review checklist.

The goal is to make semantic review consistent and fast: a reviewer should be able
to open this document, scan the checklist, and flag any conceptual-naming problems
without having to reconstruct the migration model from scratch.

---

## Quick Reference: Correct Term by Semantic Role

| If the code refers to…                   | Use this term   | Not this term    |
|------------------------------------------|-----------------|------------------|
| A schedulable execution unit             | `Task`          | `Thread`, `process` |
| A virtual memory / address-space object  | `Space`         | `address space`, `aspace`, `Process` |
| A lifecycle/accounting container         | `Job`           | `Process`        |
| A permission / credential context        | `Authority`     | `Process`, `uid/gid blob` |
| A world/filesystem context (cwd, namespace) | `Place`      | `Process.cwd`, `env` |
| A coordination domain (group, session)   | `Group`         | `pgid`, `session` |
| A resource reference (open file, etc.)   | `Handle` / FD   | `fd` (when used as a concept name in new types) |
| An inter-task notification               | `Message` / Event | `signal` (in new code) |

---

## PR Review Checklist

Use the following checklist when reviewing code that adds or modifies types,
functions, syscall handlers, procfs paths, IPC code, or documentation.

### 1. Naming audit

- [ ] Does any new **public** type or function name contain `Thread` where `Task`
      is correct? (Allowed only inside `kernel/src/task/` internals and
      compatibility layers.)
- [ ] Does any new **public** type or function name contain `Process` as a concept
      name? (Allowed only in compatibility layers and explicit Unix-compat references;
      must be annotated `// LEGACY COMPAT`.)
- [ ] Is `fork` or `Fork` introduced anywhere in new kernel model code?
      (Never permitted; use `spawn` + `exec`.)
- [ ] Is `pid` used as a **concept name** in new canonical types or docs?
      (Numeric `pid` fields at low-level kernel layer are OK; new canonical types
      must use `TaskId` or `JobId`.)
- [ ] Does new code add state to `Process` directly (rather than to a named
      subdivision like `ProcessLifecycle` or `ProcessAddressSpace`)?

### 2. Concept usage audit

- [ ] Does new execution or scheduling code use `Task` as the canonical unit?
- [ ] Does new VM / memory code route through `ProcessAddressSpace` (the `Space`
      extraction seam) rather than directly accessing `Process.mappings`?
- [ ] Does new lifecycle / accounting code route through `ProcessLifecycle` (the
      `Job` extraction seam) rather than directly accessing top-level `Process`
      lifecycle fields?
- [ ] Does new credential / permission code route through
      `kernel::authority::bridge` rather than reading `Process` fields directly?
- [ ] Does new cwd / namespace code route through `kernel::place::bridge` rather
      than reading `Process.cwd` / `Process.namespace` directly?
- [ ] Does new group / session code route through `kernel::group::bridge` rather
      than reading `Process.pgid` / `Process.sid` directly?

### 3. Bridge usage audit

- [ ] Does new code that produces a public-facing representation of a Task, Job,
      Group, Authority, or Place call the appropriate bridge module?
- [ ] Is any bridge bypassed in favor of reading `Process` or `Thread<R>` fields
      directly at a public surface?
- [ ] If a new bridge is introduced, does it follow the conventions in
      `docs/migration/bridge_architecture.md`?
- [ ] Does new bridge code include `// PROVISIONAL:` comments for mappings that
      rely on transitional state?

### 4. Unresolved mapping audit

- [ ] Does the change make assumptions about an unresolved mapping (FD/Handle,
      Signal/Message, Port/Inbox, Presence)? If so, is the assumption documented
      or does it need an explicit open-question comment?
- [ ] Does the change prematurely commit to a mapping that is still marked
      **Transitional** or **Unresolved** in `concept-mapping.md`?

### 5. Documentation and annotation audit

- [ ] Does new code that uses transitional terms include a `// LEGACY COMPAT:` or
      `// PROVISIONAL:` annotation?
- [ ] Does the change require updating `docs/migration/concept-mapping.md`
      (e.g., a mapping moved from Transitional → Stable, or a new open question
      was resolved)?
- [ ] Does the change require updating `docs/migration/process_responsibility_map.md`
      (e.g., a field moved to a new subdivision or a bridge was introduced)?

---

## Escalation: When to Block a PR

Block (request changes) if any of the following are true:

1. **New `Thread`-named public type** outside `kernel/src/task/` internals —
   rename to `Task`.
2. **New `Process`-named type** in non-compatibility code — decompose to the
   appropriate first-class concepts.
3. **New state added directly to top-level `Process`** (outside a named
   subdivision) — redirect to `ProcessLifecycle`, `ProcessAddressSpace`, or
   the appropriate bridge.
4. **Bridge bypassed** at a public surface — require the bridge to be called.
5. **`fork` / `SYS_FORK` introduced** anywhere — reject; use spawn + exec.
6. **Unresolved mapping assumed as stable** without annotation — require
   documentation of the assumption.

---

## Guidance: When to Allow Legacy Terms

Legacy terms are permitted in these contexts without blocking the PR:

| Context | Permitted legacy term | Required annotation |
|---------|-----------------------|---------------------|
| `kernel/src/task/` internal struct fields | `Thread<R>`, `ThreadState` | None required (internal only) |
| POSIX compatibility layer | `process`, `pid`, `signal`, `fork` semantics described | `// LEGACY COMPAT` |
| Bridge module internals | `Process`, `Thread` as input types | `// PROVISIONAL:` comment on each mapping |
| Low-level kernel numeric fields (`pid: u32`) | `pid` | Inline comment noting future split |
| Comments explaining what is *not* present | `fork`, `SYS_FORK` | None required |
| This document and concept-mapping.md | All legacy terms | None required (definitional use) |

---

## Migration Phase Reference

PRs are reviewed against the current migration phase. The phase determines which
concepts are considered stable vs. transitional.

| Phase | Concepts considered stable | Notes |
|-------|---------------------------|-------|
| Pre-3 | `Task`, `TaskState`       | `Thread` is the internal backing |
| 3     | `Task`, `Job`, `JobExit`, `JobWaitResult` | `Process` lifecycle bridged |
| 4     | + `Group`                 | `pgid`/`sid` bridged |
| 7     | + `Authority`             | `exec_path`/`name` bridged |
| 8     | + `Place`                 | `cwd`/`namespace` bridged |
| 9+    | + `Space` (target)        | `ProcessAddressSpace` is extraction seam |
|       | + Handle table (target)   | `fd_table` quarantined until Phase 9+ |

---

## Related Documents

- `docs/migration/concept-mapping.md` — the canonical lexicon this checklist is derived from
- `docs/migration/process_responsibility_map.md` — field-level decomposition and extraction sequencing
- `docs/migration/bridge_architecture.md` — bridge layer design, conventions, and guardrails
- `docs/concepts/thingos-guardrails.md` — architecture guardrails (spawn+exec, VFS-first, etc.)
- `.github/PULL_REQUEST_TEMPLATE.md` — machine-readable PR checklist (references this document)
