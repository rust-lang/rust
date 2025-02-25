# rust-analyzer

At its core, rust-analyzer is a **library** for semantic analysis of
Rust code as it changes over time. This manual focuses on a specific
usage of the library -- running it as part of a server that implements
the [Language Server
Protocol](https://microsoft.github.io/language-server-protocol/) (LSP).
The LSP allows various code editors, like VS Code, Emacs or Vim, to
implement semantic features like completion or goto definition by
talking to an external language server process.

To improve this document, send a pull request:
[https://github.com/rust-lang/rust-analyzer](https://github.com/rust-lang/rust-analyzer/blob/master/docs/book/README.md)

The manual is written in markdown and includes
some extra files which are generated from the source code. Run
`cargo test` and `cargo xtask codegen` to create these.

If you have questions about using rust-analyzer, please ask them in the
["IDEs and Editors"](https://users.rust-lang.org/c/ide/14) topic of Rust
users forum.
