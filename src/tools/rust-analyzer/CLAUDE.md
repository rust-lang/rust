## AI Policy

Follow `AI_POLICY.md`. In particular:

- Do not use AI to author issue/PR comments or replies to maintainers.
- Do not autonomously open issues or pull requests.
- Do not author code for issues labeled both `E-easy` and `E-has-instructions`.
- The human contributor must understand the changes and disclose AI use as required by the policy.


## Repository Guides

- Architecture and crate ownership: `docs/book/src/contributing/architecture.md`
- Rust style: `docs/book/src/contributing/style.md`
- Testing conventions and fixture syntax: `docs/book/src/contributing/testing.md`
- Contributor workflows: `docs/book/src/contributing/README.md`
- AI restrictions: `AI_POLICY.md`

Read the relevant sections before making architectural, generated-code, protocol, or test-harness changes.

## Change Workflow

- Find the nearest existing implementation and its tests before adding new code.
- Extend existing helpers and test harnesses instead of creating parallel abstractions.
- Keep changes focused and prefer the smallest change that fits the existing design.
- When fixing a bug, add the smallest fixture that reproduces it and test the behavior through the existing interface.

## Scope and Dependencies

- Treat new `pub` items, public re-exports, and Cargo dependencies as architectural changes, not routine implementation details.
- Prefer keeping functionality inside the crate that owns the relevant data.
- Be conservative with crates.io dependencies. Reuse existing dependencies or `stdx`; do not add small helper crates without strong justification.

## Key Invariants

- User-provided Rust code, malformed syntax, broken builds, and proc-macro failures must not cause ordinary IDE features to panic.
- Assert invariants liberally. For impossible conditions from which the server can recover, prefer `stdx::never!` or `stdx::always!` and return a safe fallback instead of panicking.

## Testing

- Many feature tests use Rust-code fixtures and `expect-test` snapshots. Follow the nearest existing test helper and fixture convention rather than introducing a new test harness.
- Before planning or writing fixture-based tests, review `docs/book/src/contributing/testing.md` for fixture annotations, `minicore`, and multi-file/multi-crate syntax.
- Keep Rust fixtures minimal; remove syntax unrelated to the behavior under test.
- Use unindented multiline raw strings, matching nearby tests.
- For regressions, first reproduce the failure with a focused test, then implement the fix.

## Generated Code

- Generated files are committed. Edit the generator rather than generated output.
- Run `cargo xtask codegen` after changing grammar, generated AST definitions, configuration schemas, or other codegen inputs.
- After adding parser inline tests (`// test name`), run `cargo test -p xtask`, update the relevant expectations, and inspect the generated diff.

## Validation

- Start with the narrowest relevant test: `cargo test -p <crate> <test-name>` or `cargo test -p <crate>`.
- After Rust changes, run the affected crate's tests and `cargo clippy -p <crate> --all-targets -- --cap-lints warn`; broaden validation when the change crosses crates.
- Use `cargo lint` to run Clippy on all workspace targets.
- Run `cargo xtask tidy` for repository-wide structural or generated-code changes.
- When updating snapshots with `UPDATE_EXPECT=1`, inspect the expectation diff rather than accepting it blindly.
- Use `RUN_SLOW_TESTS=1 cargo test` when the affected area has slow tests.
