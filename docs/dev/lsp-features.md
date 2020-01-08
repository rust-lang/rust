# Supported LSP features

This list documents LSP features, supported by rust-analyzer.

## General
- [x] [initialize](https://microsoft.github.io/language-server-protocol/specification#initialize)
- [x] [initialized](https://microsoft.github.io/language-server-protocol/specification#initialized)
- [x] [shutdown](https://microsoft.github.io/language-server-protocol/specification#shutdown)
- [ ] [exit](https://microsoft.github.io/language-server-protocol/specification#exit)
- [x] [$/cancelRequest](https://microsoft.github.io/language-server-protocol/specification#cancelRequest)

## Workspace
- [ ] [workspace/workspaceFolders](https://microsoft.github.io/language-server-protocol/specification#workspace_workspaceFolders)
- [ ] [workspace/didChangeWorkspaceFolders](https://microsoft.github.io/language-server-protocol/specification#workspace_didChangeWorkspaceFolders)
- [x] [workspace/didChangeConfiguration](https://microsoft.github.io/language-server-protocol/specification#workspace_didChangeConfiguration)
- [ ] [workspace/configuration](https://microsoft.github.io/language-server-protocol/specification#workspace_configuration)
- [x] [workspace/didChangeWatchedFiles](https://microsoft.github.io/language-server-protocol/specification#workspace_didChangeWatchedFiles)
- [x] [workspace/symbol](https://microsoft.github.io/language-server-protocol/specification#workspace_symbol)
- [ ] [workspace/applyEdit](https://microsoft.github.io/language-server-protocol/specification#workspace_applyEdit)

## Text Synchronization
- [x] [textDocument/didOpen](https://microsoft.github.io/language-server-protocol/specification#textDocument_didOpen)
- [x] [textDocument/didChange](https://microsoft.github.io/language-server-protocol/specification#textDocument_didChange)
- [ ] [textDocument/willSave](https://microsoft.github.io/language-server-protocol/specification#textDocument_willSave)
- [ ] [textDocument/willSaveWaitUntil](https://microsoft.github.io/language-server-protocol/specification#textDocument_willSaveWaitUntil)
- [x] [textDocument/didSave](https://microsoft.github.io/language-server-protocol/specification#textDocument_didSave)
- [x] [textDocument/didClose](https://microsoft.github.io/language-server-protocol/specification#textDocument_didClose)

## Diagnostics
- [x] [textDocument/publishDiagnostics](https://microsoft.github.io/language-server-protocol/specification#textDocument_publishDiagnostics)

## Lanuguage Features
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
 - trigger characters: `(`,  `,`
- [ ] [textDocument/declaration](https://microsoft.github.io/language-server-protocol/specification#textDocument_declaration)
- [x] [textDocument/definition](https://microsoft.github.io/language-server-protocol/specification#textDocument_definition)
- [x] [textDocument/typeDefinition](https://microsoft.github.io/language-server-protocol/specification#textDocument_typeDefinition)
- [x] [textDocument/implementation](https://microsoft.github.io/language-server-protocol/specification#textDocument_implementation)
- [x] [textDocument/references](https://microsoft.github.io/language-server-protocol/specification#textDocument_references)
- [x] [textDocument/documentHighlight](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentHighlight)
- [x] [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specification#textDocument_documentSymbol)
- [x] [textDocument/codeAction](https://microsoft.github.io/language-server-protocol/specification#textDocument_codeAction)
- [x] [textDocument/selectionRange](https://github.com/Microsoft/language-server-protocol/issues/613)
 - rust-analyzer.syntaxTree
 - rust-analyzer.matchingBrace
 - rust-analyzer.parentModule
 - rust-analyzer.joinLines
 - rust-analyzer.run
 - rust-analyzer.analyzerStatus
- [x] [textDocument/codeLens](https://microsoft.github.io/language-server-protocol/specification#textDocument_codeLens)
- [x] [codeLens/resolve](https://microsoft.github.io/language-server-protocol/specification#codeLens_resolve)
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
