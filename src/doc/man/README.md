# Generating the `rustc(1)` Man Page

This document describes how the `rustc(1)` man page is generated, where the generation logic lives, and how it reliably targets the correct `rustc` binary regardless of the working directory.

## Goal

Ensure the `rustc(1)` man page is generated **entirely from `rustc --help -v` output** so that the documentation reflects the compiler’s actual behavior and does not drift over time.

## Source of Truth

The man page is generated from:

```
rustc --help -v
```

This output is treated as authoritative. No hand-written sections are maintained.

## Tooling

 `help2man` to generate the man page:

* `help2man` parses CLI help output
* `--help -v` is passed explicitly to include verbose options

## Script Location

The generation logic lives alongside the artifact it produces:

This keeps documentation tooling local and avoids coupling it to the main build system.

## Generation Script

`generate-rustc-man.sh` is the canonical way to regenerate the man page.

* The script resolves its own location to find the repository root
* The path to `rustc` is passed explicitly via an environment variable
* No relative assumptions are made about execution location

### Script Behavior

* Resolve repository root via the script’s directory
* Resolve the `rustc` binary path (provided by the build system or caller)
* Invoke `help2man` with an explicit binary path
* Write output deterministically to `src/doc/man/rustc.1`

The build or release tooling invoking this script **must pass the correct `rustc` binary** (e.g., the freshly built compiler) via environment or explicit path.

## Regenerating the Man Page

From the repository root:

```
RUSTC=path/to/rustc ./src/doc/man/generate-rustc-man.sh
```

This guarantees:

* The correct compiler is documented
* Output is written to the correct location

## Rationale

* Keeps the man page in sync with real compiler behavior
* Avoids manual duplication of CLI documentation
* Makes regeneration explicit and reproducible 
