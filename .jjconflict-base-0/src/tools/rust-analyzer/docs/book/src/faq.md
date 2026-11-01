# Troubleshooting FAQ

### I see a warning "Variable `None` should have snake_case name, e.g. `none`"

rust-analyzer fails to resolve `None`, and thinks you are binding to a variable
named `None`. That's usually a sign of a corrupted sysroot. Try removing and re-installing
it: `rustup component remove rust-src` then `rustup component add rust-src`.

### Rust Analyzer and Cargo compete over the build lock

Rust Analyzer invokes Cargo in the background, and it can thus block manually executed
`cargo` commands from making progress (or vice-versa). In some cases, this can also cause
unnecessary recompilations caused by cache thrashing. To avoid this, you can configure
Rust Analyzer to use a [different target directory](./configuration.md#cargo.targetDir).
This will allow both the IDE and Cargo to make progress independently, at the cost of
increased disk space usage caused by the duplicated artifact directories.
