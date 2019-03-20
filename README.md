# Rust Analyzer

[![Build Status](https://travis-ci.org/rust-analyzer/rust-analyzer.svg?branch=master)](https://travis-ci.org/rust-analyzer/rust-analyzer)

Rust Analyzer is an **experimental** modular compiler frontend for the Rust
language. It is a part of a larger rls-2.0 effort to create excellent IDE
support for Rust. If you want to get involved, check the rls-2.0 working group
in the compiler-team repository:

https://github.com/rust-lang/compiler-team/tree/master/working-groups/rls-2.0

Work on the Rust Analyzer is sponsored by

[![Ferrous Systems](https://ferrous-systems.com/images/ferrous-logo-text.svg)](https://ferrous-systems.com/)

## Quick Start

Rust analyzer builds on Rust >= 1.31.0 and uses the 2018 edition.

```
# run tests
$ cargo test

# show syntax tree of a Rust file
$ cargo run --package ra_cli parse < crates/ra_syntax/src/lib.rs

# show symbols of a Rust file
$ cargo run --package ra_cli symbols < crates/ra_syntax/src/lib.rs

# install the language server
$ cargo install-lsp
or
$ cargo install --path crates/ra_lsp_server
```

See [these instructions](./editors/README.md) for VS Code setup and the list of
features (some of which are VS Code specific).

## Debugging

See [these instructions](./DEBUGGING.md) on how to debug the vscode extension and the lsp server.

## Getting in touch

We are on the rust-lang Zulip!

https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Frls-2.2E0

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [ARCHITECTURE.md](./ARCHITECTURE.md)


## License

Rust analyzer is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0).

See LICENSE-APACHE and LICENSE-MIT for details.
