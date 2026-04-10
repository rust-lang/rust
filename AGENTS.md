# Agent Instructions

## Rust Engineering Standards

When making Rust changes, prefer correctness, clarity, and maintainability over cleverness.

### Idiomatic Rust

- Write code that is straightforward to read and follows standard Rust conventions.
- Prefer iterators, pattern matching, and enums over ad-hoc state flags and complex control flow.
- Minimize cloning and allocations; pass references where practical and use ownership intentionally.
- Keep functions focused and small; extract helpers when logic becomes hard to scan.
- Avoid panics in non-test code unless truly unrecoverable; return structured errors instead.

### Safety and Best Practices

- Do not use `unwrap()`/`expect()` in production paths when errors can be handled or propagated.
- Keep `unsafe` blocks minimal, documented with invariants, and covered by tests.
- Prefer explicit types when inference hurts readability.
- Use existing abstractions in the codebase before introducing new patterns or dependencies.
- Preserve backward compatibility and existing behavior unless the issue explicitly requires change.

### Testing Expectations

- Add or update tests for behavior changes and bug fixes.
- Prefer targeted tests first (unit or focused integration tests), then broader suites as needed.
- Include negative/error-path tests for new logic where applicable.
- Avoid brittle tests; assert semantics, not incidental formatting or implementation details.

### Code Quality Gates

Run relevant checks before finishing work (scope to changed areas when possible):

```bash
./x.py fmt
./x.py check
./x.py test <path-or-suite>
```

If your change touches compiler behavior, add or update the appropriate `tests/ui`, `tests/codegen`, or backend-specific tests.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
