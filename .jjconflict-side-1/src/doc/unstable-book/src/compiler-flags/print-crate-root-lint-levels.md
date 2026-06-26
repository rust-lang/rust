# `print=crate-root-lint-levels`

The tracking issue for this feature is: [#139180](https://github.com/rust-lang/rust/issues/139180).

------------------------

This option of the `--print` flag print the list of lints with print out all the lints and their associated levels (`allow`, `warn`, `deny`, `forbid`) based on the regular Rust rules at crate root, that is *(roughly)*:
 - command line args (`-W`, `-A`, `--force-warn`, `--cap-lints`, ...)
 - crate root attributes (`#![allow]`, `#![warn]`, `#[expect]`, ...)
 - *the special `warnings` lint group*
 - the default lint level

The output format is `LINT_NAME=LINT_LEVEL`, e.g.:
```text
unknown_lint=warn
arithmetic_overflow=deny
```

To be used like this:

```bash
rustc --print=crate-root-lint-levels -Zunstable-options lib.rs
```
