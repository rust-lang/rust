# Diagnostics

While most errors and warnings provided by rust-analyzer come from the
`cargo check` integration, there’s a growing number of diagnostics
implemented using rust-analyzer’s own analysis. Some of these
diagnostics don’t respect `#[allow]` or `#[deny]` attributes yet, but
can be turned off using the `rust-analyzer.diagnostics.enable`,
`rust-analyzer.diagnostics.experimental.enable` or
`rust-analyzer.diagnostics.disabled` settings.

## Clippy

To run `cargo clippy` instead of `cargo check`, you can set
`"rust-analyzer.check.command": "clippy"`.

{{#include diagnostics_generated.md:2:}}
