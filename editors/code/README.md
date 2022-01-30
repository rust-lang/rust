# rust-analyzer

Provides support for rust-analyzer: novel LSP server for the Rust programming language.

**Note** the extension may cause conflicts with the official Rust extension. It is recommended to disable the Rust extension when using the rust-analyzer extension.

## Sponsor

Work on rust-analyzer is sponsored by

[<img src="https://user-images.githubusercontent.com/1711539/58105231-cf306900-7bee-11e9-83d8-9f1102e59d29.png" alt="Ferrous Systems" width="300">](https://ferrous-systems.com/)

- [Mozilla](https://www.mozilla.org/en-US/)
- [Embark Studios](https://embark-studios.com/)

If you want to sponsor:

- [OpenCollective](https://opencollective.com/rust-analyzer/)
- [Github Sponsors](https://github.com/sponsors/rust-analyzer)

## Features

- [code completion], [imports insertion]
- [go to definition], [implementation], [type definition]
- [find all references], [workspace symbol search], [rename]
- [types and documentation on hover]
- [inlay hints]
- [semantic syntax highlighting]
- a lot of [assist(code actions)]
- apply suggestions from errors
- ... and many more, checkout the [manual] to see them all

[code completion]: https://rust-analyzer.github.io/manual.html#magic-completions
[imports insertion]: https://rust-analyzer.github.io/manual.html#auto-import
[go to definition]: https://rust-analyzer.github.io/manual.html#go-to-definition
[implementation]: https://rust-analyzer.github.io/manual.html#go-to-implementation
[type definition]: https://rust-analyzer.github.io/manual.html#go-to-type-definition
[find all references]: https://rust-analyzer.github.io/manual.html#find-all-references
[workspace symbol search]: https://rust-analyzer.github.io/manual.html#workspace-symbol
[rename]: https://rust-analyzer.github.io/manual.html#rename
[types and documentation on hover]: https://rust-analyzer.github.io/manual.html#hover
[inlay hints]: https://rust-analyzer.github.io/manual.html#inlay-hints
[semantic syntax highlighting]: https://rust-analyzer.github.io/manual.html#semantic-syntax-highlighting
[assist(code actions)]: https://rust-analyzer.github.io/manual.html#assists-code-actions

[manual]: https://rust-analyzer.github.io/manual.html

## Quick start

1. Install [rustup]
2. Install the [rust-analyzer extension]

[rustup]: https://rustup.rs
[rust-analyzer extension]: https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer

## Configuration

This extension provides configurations through VSCode's configuration settings. All the configurations are under `rust-analyzer.*`.

See <https://rust-analyzer.github.io/manual.html#vs-code-2> for more information on VSCode specific configurations.

## Communication

For usage and troubleshooting requests, please use "IDEs and Editors" category of the Rust forum:

<https://users.rust-lang.org/c/ide/14>

## Documentation

See <https://rust-analyzer.github.io/> for more information.
