<p align="center">
  <img
    src="https://raw.githubusercontent.com/rust-lang/rust-analyzer/master/assets/logo-wide.svg"
    alt="rust-analyzer logo">
</p>

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
in the manual.

## Quick Start

https://rust-analyzer.github.io/book/installation.html

## Documentation

If you want to **contribute** to rust-analyzer check out the [CONTRIBUTING.md](./CONTRIBUTING.md) or
if you are just curious about how things work under the hood, see the
[Contributing](https://rust-analyzer.github.io/book/contributing) section of the manual.

If you want to **use** rust-analyzer's language server with your editor of
choice, check [the manual](https://rust-analyzer.github.io/book/).
It also contains some tips & tricks to help you be more productive when using rust-analyzer.

## Security and Privacy

See the [security](https://rust-analyzer.github.io/book/security.html) and
[privacy](https://rust-analyzer.github.io/book/privacy.html) sections of the manual.

## Communication

For usage and troubleshooting requests, please use "IDEs and Editors" category of the Rust forum:

https://users.rust-lang.org/c/ide/14

For questions about development and implementation, join rust-analyzer working group on Zulip:

https://rust-lang.zulipchat.com/#narrow/stream/185405-t-compiler.2Frust-analyzer

## Quick Links

* Website: https://rust-analyzer.github.io/
* Metrics: https://rust-analyzer.github.io/metrics/
* API docs: https://rust-lang.github.io/rust-analyzer/ide/
* Changelog: https://rust-analyzer.github.io/thisweek

## License

rust-analyzer is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.
