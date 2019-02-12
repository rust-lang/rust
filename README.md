# Rust Analyzer

[![Build Status](https://travis-ci.org/rust-analyzer/rust-analyzer.svg?branch=master)](https://travis-ci.org/rust-analyzer/rust-analyzer)

Rust Analyzer is an **experimental** modular compiler frontend for the Rust
language. It is a part of a larger rls-2.0 effort to create excellent IDE
support for Rust. If you want to get involved, check rls-2.0 working group repository:

https://github.com/rust-analyzer/WG-rls2.0

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
- [x] [textDocument/implementation](https://microsoft.github.io/language-server-protocol/specification#textDocument_implementation)
- [x] [textDocument/references](https://microsoft.github.io/language-server-protocol/specification#textDocument_references)
- [x] [textDocument/documentHighlight](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentHighlight)
- [x] [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentSymbol)
- [x] [textDocument/codeAction](https://microsoft.github.io/language-server-protocol/specification#textDocument_codeAction)
 - rust-analyzer.syntaxTree
 - rust-analyzer.extendSelection
 - rust-analyzer.matchingBrace
 - rust-analyzer.parentModule
 - rust-analyzer.joinLines
 - rust-analyzer.run
 - rust-analyzer.analyzerStatus
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

## License

Rust analyzer is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0).

See LICENSE-APACHE and LICENSE-MIT for details.
