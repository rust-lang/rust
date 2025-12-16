# Generating the `rustc(1)` Man Page

This document describes how the `rustc(1)` man page is generated, where the logic lives, and how it robustly targets the correct `rustc` binary regardless of execution context.

## Goal

Ensure the `rustc(1)` man page is generated **entirely from `rustc --help -v` output** so that documentation reflects the real compiler and does not drift.

## Source of Truth

The man page is generated from:

```
rustc --help -v
```

This output is treated as authoritative. No manual sections are maintained.

## Tooling

- [`help2man`](https://www.gnu.org/software/help2man/) to generate the man page:
  - Parses CLI help output (`rustc --help -v`)
  - Provides a formatted groff man page

## Script Location

- All logic and artifacts are in `src/doc/man`.
- This avoids coupling to the main build system.

## Generation Script

`generate-rustc-man.sh` is the reproducible way to regenerate the man page.

- The script resolves its own location and output directory.
- The path to the *actual* `rustc` binary is passed as an argument.
- No assumptions made about the working directory.

### Script Behavior

- Resolves absolute path to provided `rustc` binary.
- Runs `help2man` with `--help -v` on that binary.
- Writes output to `src/doc/man/rustc.1`.

The build or release tooling invoking this script **must pass the correct `rustc` binary** (e.g., a freshly built compiler).

## Regenerating the Man Page

From the repository root *after building rustc*:

```bash
./src/doc/man/generate-rustc-man.sh path/to/rustc
```

This guarantees:
- The correct (built) compiler is documented
- Output always lands in the right place

## Prerequisites

- `help2man` must be installed and in your `$PATH`.

On most Linux distros: `sudo apt install help2man` (Debian/Ubuntu) or `sudo dnf install help2man` (Fedora).

## Rationale

- Keeps the man page in sync with real compiler behavior
- Avoids manual duplication of CLI documentation
- Regeneration is explicit and reproducible

## Notes

- The generated `rustc.1` can be `.gitignore`d as desired, depending on release/build process.
- If you change how `rustc` outputs help, rerun this process to update the man page
