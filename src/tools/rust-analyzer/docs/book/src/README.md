# rust-analyzer

rust-analyzer is a language server that provides IDE functionality for
writing Rust programs. You can use it with any editor that supports
the [Language Server
Protocol](https://microsoft.github.io/language-server-protocol/) (VS
Code, Vim, Emacs, Zed, etc).

rust-analyzer features include go-to-definition, find-all-references,
refactorings and code completion. rust-analyzer also supports
integrated formatting (with rustfmt) and integrated diagnostics (with
rustc and clippy).

Internally, rust-analyzer is structured as a set of libraries for
analyzing Rust code. See
[Architecture](https://rust-analyzer.github.io/book/contributing/architecture.html)
for more details.

To improve this document, send a pull request:
[https://github.com/rust-lang/rust-analyzer](https://github.com/rust-lang/rust-analyzer/blob/master/docs/book/README.md)

The manual is written in markdown and includes
some extra files which are generated from the source code. Run
`cargo test` and `cargo xtask codegen` to create these.

If you have questions about using rust-analyzer, please ask them in the
["IDEs and Editors"](https://users.rust-lang.org/c/ide/14) topic of Rust
users forum.
