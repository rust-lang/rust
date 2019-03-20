The main interface to rust-analyzer is the
[LSP](https://microsoft.github.io/language-server-protocol/) implementation. To
install lsp server, use `cargo install-lsp`, which is a shorthand for `cargo
install --package ra_lsp_server`. The binary is named `ra_lsp_server`, you
should be able to use it with any LSP-compatible editor. We use custom
extensions to LSP, so special client-side support is required to take full
advantage of rust-analyzer. This repository contains support code for VS Code
and Emacs.

Rust Analyzer needs sources of rust standard library to work, so you might need
to execute

```
$ rustup component add rust-src
```

See [./features.md](./features.md) document for a list of features that are available.

## VS Code

Prerequisites:

In order to build the VS Code plugin, you need to have node.js and npm with
a minimum version of 10 installed. Please refer to
[node.js and npm documentation](https://nodejs.org) for installation instructions.

You will also need the most recent version of VS Code: we don't try to
maintain compatibility with older versions yet.

The experimental VS Code plugin can then be built and installed by executing the
following commands:

```
$ git clone https://github.com/rust-analyzer/rust-analyzer.git --depth 1
$ cd rust-analyzer
$ cargo install-code
```

This will run `cargo install --package ra_lsp_server` to install the server
binary into `~/.cargo/bin`, and then will build and install plugin from
`editors/code`. See
[this](https://github.com/rust-analyzer/rust-analyzer/blob/69ee5c9c5ef212f7911028c9ddf581559e6565c3/crates/tools/src/main.rs#L37-L56)
for details. The installation is expected to *just work*, if it doesn't, report
bugs!

It's better to remove existing Rust plugins to avoid interference.

Beyond basic LSP features, there are some extension commands which you can
invoke via <kbd>Ctrl+Shift+P</kbd> or bind to a shortcut. See [./features.md](./features.md)
for details.

### Settings

* `rust-analyzer.highlightingOn`: enables experimental syntax highlighting
* `rust-analyzer.showWorkspaceLoadedNotification`: to ease troubleshooting, a
  notification is shown by default when a workspace is loaded
* `rust-analyzer.enableEnhancedTyping`: by default, rust-analyzer intercepts
  `Enter` key to make it easier to continue comments
* `rust-analyzer.raLspServerPath`: path to `ra_lsp_server` executable
* `rust-analyzer.enableCargoWatchOnStartup`: prompt to install & enable `cargo
  watch` for live error highlighting (note, this **does not** use rust-analyzer)
* `rust-analyzer.trace.server`: enables internal logging


## Emacs

Prerequisites:

`emacs-lsp`, `dash` and `ht` packages.

Installation:

* add
[ra-emacs-lsp.el](https://github.com/rust-analyzer/rust-analyzer/blob/69ee5c9c5ef212f7911028c9ddf581559e6565c3/editors/emacs/ra-emacs-lsp.el)
to load path and require it in `init.el`
* run `lsp` in a rust buffer
* (Optionally) bind commands like `rust-analyzer-join-lines` or `rust-analyzer-extend-selection` to keys
