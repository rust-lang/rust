# Rust Analyzer

[![Build Status](https://travis-ci.org/rust-analyzer/rust-analyzer.svg?branch=master)](https://travis-ci.org/rust-analyzer/rust-analyzer)

Rust Analyzer is an **experimental** modular compiler frontend for the Rust
language, which aims to lay a foundation for excellent IDE support.

It doesn't implement much of compiler functionality yet, but the white-space
preserving Rust parser works, and there are significant chunks of overall
architecture (indexing, on-demand & lazy computation, snapshotable world view)
in place. Some basic IDE functionality is provided via a language server.

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
$ cargo install --path crates/ra_lsp_server
```

See [these instructions](./editors/README.md) for VS Code setup and the list of
features (some of which are VS Code specific).

## Debugging

See [these instructions](./DEBUGGING.md) on how to debug the vscode extension and the lsp server.

## Current Status and Plans

Rust analyzer aims to fill the same niche as the official [Rust Language
Server](https://github.com/rust-lang-nursery/rls), but uses a significantly
different architecture. More details can be found [in this
thread](https://internals.rust-lang.org/t/2019-strategy-for-rustc-and-the-rls/8361),
but the core issue is that RLS works in the "wait until user stops typing, run
the build process, save the results of the analysis" mode, which arguably is the
wrong foundation for IDE.

Rust Analyzer is an experimental project at the moment, there's exactly zero
guarantees that it becomes production-ready one day.

The near/mid term plan is to work independently of the main rustc compiler and
implement at least simplistic versions of name resolution, macro expansion and
type inference. The purpose is two fold:

- to quickly bootstrap usable and useful language server: solution that covers
  80% of Rust code will be useful for IDEs, and will be vastly simpler than 100%
  solution.

- to understand how the consumer-side of compiler API should look like
  (especially it's on-demand aspects). If you have `get_expression_type`
  function, you can write a ton of purely-IDE features on top of it, even if the
  function is only partially correct. Pluging in the precise function afterwards
  should just make IDE features more reliable.

The long term plan is to merge with the mainline rustc compiler, probably around
the HIR boundary? That is, use rust analyzer for parsing, macro expansion and
related bits of name resolution, but leave the rest (including type inference
and trait selection) to the existing rustc.

## Supported LSP features

### General
- [x] [initialize](https://microsoft.github.io/language-server-protocol/specification#initialize)
- [x] [initialized](https://microsoft.github.io/language-server-protocol/specification#initialized)
- [x] [shutdown](https://microsoft.github.io/language-server-protocol/specification#shutdown)
- [ ] [exit](https://microsoft.github.io/language-server-protocol/specification#exit)
- [x] [$/cancelRequest](https://microsoft.github.io/language-server-protocol/specification#cancelRequest)

### Workspace
- [ ] [workspace/workspaceFolders](https://microsoft.github.io/language-server-protocol/specification#workspace_workspaceFolders)
- [ ] [workspace/didChangeWorkspaceFolders](https://microsoft.github.io/language-server-protocol/specification#workspace_didChangeWorkspaceFolders)
- [x] [workspace/didChangeConfiguration](https://microsoft.github.io/language-server-protocol/specification#workspace_didChangeConfiguration)
- [ ] [workspace/configuration](https://microsoft.github.io/language-server-protocol/specification#workspace_configuration)
- [x] [workspace/didChangeWatchedFiles](https://microsoft.github.io/language-server-protocol/specification#workspace_didChangeWatchedFiles)
- [x] [workspace/symbol](https://microsoft.github.io/language-server-protocol/specification#workspace_symbol)
- [x] [workspace/executeCommand](https://microsoft.github.io/language-server-protocol/specification#workspace_executeCommand)
 - `apply_code_action`
- [ ] [workspace/applyEdit](https://microsoft.github.io/language-server-protocol/specification#workspace_applyEdit)

### Text Synchronization
- [x] [textDocument/didOpen](https://microsoft.github.io/language-server-protocol/specification#textDocument_didOpen)
- [x] [textDocument/didChange](https://microsoft.github.io/language-server-protocol/specification#textDocument_didChange)
- [ ] [textDocument/willSave](https://microsoft.github.io/language-server-protocol/specification#textDocument_willSave)
- [ ] [textDocument/willSaveWaitUntil](https://microsoft.github.io/language-server-protocol/specification#textDocument_willSaveWaitUntil)
- [x] [textDocument/didSave](https://microsoft.github.io/language-server-protocol/specification#textDocument_didSave)
- [x] [textDocument/didClose](https://microsoft.github.io/language-server-protocol/specification#textDocument_didClose)

### Diagnostics
- [x] [textDocument/publishDiagnostics](https://microsoft.github.io/language-server-protocol/specification#textDocument_publishDiagnostics)

### Lanuguage Features
- [x] [textDocument/completion](https://microsoft.github.io/language-server-protocol/specification#textDocument_completion)
 - open close: false
 - change: Full
 - will save: false
 - will save wait until: false
 - save: false
- [x] [completionItem/resolve](https://microsoft.github.io/language-server-protocol/specification#completionItem_resolve)
 - resolve provider: none
 - trigger characters: `:`, `.`
- [x] [textDocument/hover](https://microsoft.github.io/language-server-protocol/specification#textDocument_hover)
- [x] [textDocument/signatureHelp](https://microsoft.github.io/language-server-protocol/specification#textDocument_signatureHelp)
 - trigger characters: `(`,  `,`,  `)`
- [ ] [textDocument/declaration](https://microsoft.github.io/language-server-protocol/specification#textDocument_declaration)
- [x] [textDocument/definition](https://microsoft.github.io/language-server-protocol/specification#textDocument_definition)
- [ ] [textDocument/typeDefinition](https://microsoft.github.io/language-server-protocol/specification#textDocument_typeDefinition)
- [ ] [textDocument/implementation](https://microsoft.github.io/language-server-protocol/specification#textDocument_implementation)
- [x] [textDocument/references](https://microsoft.github.io/language-server-protocol/specification#textDocument_references)
- [x] [textDocument/documentHighlight](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentHighlight)
- [x] [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentSymbol)
- [x] [textDocument/codeAction](https://microsoft.github.io/language-server-protocol/specification#textDocument_codeAction)
 - ra_lsp.syntaxTree
 - ra_lsp.extendSelection
 - ra_lsp.matchingBrace
 - ra_lsp.parentModule
 - ra_lsp.joinLines
 - ra_lsp.run
 - ra_lsp.analyzerStatus
- [x] [textDocument/codeLens](https://microsoft.github.io/language-server-protocol/specification#textDocument_codeLens)
- [ ] [textDocument/documentLink](https://microsoft.github.io/language-server-protocol/specification#codeLens_resolve)
- [ ] [documentLink/resolve](https://microsoft.github.io/language-server-protocol/specification#documentLink_resolve)
- [ ] [textDocument/documentColor](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentColor)
- [ ] [textDocument/colorPresentation](https://microsoft.github.io/language-server-protocol/specification#textDocument_colorPresentation)
- [x] [textDocument/formatting](https://microsoft.github.io/language-server-protocol/specification#textDocument_formatting)
- [ ] [textDocument/rangeFormatting](https://microsoft.github.io/language-server-protocol/specification#textDocument_rangeFormatting)
- [x] [textDocument/onTypeFormatting](https://microsoft.github.io/language-server-protocol/specification#textDocument_onTypeFormatting)
 - first trigger character: `=`
 - more trigger character `.`
- [x] [textDocument/rename](https://microsoft.github.io/language-server-protocol/specification#textDocument_rename)
- [x] [textDocument/prepareRename](https://microsoft.github.io/language-server-protocol/specification#textDocument_prepareRename)
- [x] [textDocument/foldingRange](https://microsoft.github.io/language-server-protocol/specification#textDocument_foldingRange)

## Getting in touch

We have a Discord server dedicated to compilers and language servers
implemented in Rust: [https://discord.gg/sx3RQZB](https://discord.gg/sx3RQZB).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [ARCHITECTURE.md](./ARCHITECTURE.md)

## License

Rust analyzer is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0).

See LICENSE-APACHE and LICENSE-MIT for details.
