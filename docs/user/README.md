The main interface to rust-analyzer is the
[LSP](https://microsoft.github.io/language-server-protocol/) implementation. To
install lsp server, use `cargo install-ra --server`, which is a shorthand for `cargo
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
$ cargo install-ra
```

The automatic installation is expected to *just work* for common cases, if it
doesn't, report bugs!

If you have an unusual setup (for example, `code` is not in the `PATH`), you
should adapt these manual installation instructions:

```
$ git clone https://github.com/rust-analyzer/rust-analyzer.git --depth 1
$ cd rust-analyzer
$ cargo install --path ./crates/ra_lsp_server/ --force
$ cd ./editors/code
$ npm install
$ ./node_modules/vsce/out/vsce package
$ code --install-extension ./ra-lsp-0.0.1.vsix
```

It's better to remove existing Rust plugins to avoid interference.

Beyond basic LSP features, there are some extension commands which you can
invoke via <kbd>Ctrl+Shift+P</kbd> or bind to a shortcut. See [./features.md](./features.md)
for details.

For updates, pull the latest changes from the master branch, run `cargo install-ra` again, and **restart** VS Code instance.
See https://github.com/microsoft/vscode/issues/72308[microsoft/vscode#72308] for why a full restart is needed.

### Settings

* `rust-analyzer.highlightingOn`: enables experimental syntax highlighting
* `rust-analyzer.showWorkspaceLoadedNotification`: to ease troubleshooting, a
  notification is shown by default when a workspace is loaded
* `rust-analyzer.enableEnhancedTyping`: by default, rust-analyzer intercepts
  `Enter` key to make it easier to continue comments. Note that it may conflict with VIM emulation plugin.
* `rust-analyzer.raLspServerPath`: path to `ra_lsp_server` executable
* `rust-analyzer.enableCargoWatchOnStartup`: prompt to install & enable `cargo
  watch` for live error highlighting (note, this **does not** use rust-analyzer)
* `rust-analyzer.cargo-watch.check-arguments`: cargo-watch check arguments.
  (e.g: `--features="shumway,pdf"` will run as `cargo watch -x "check --features="shumway,pdf""` )
* `rust-analyzer.trace.server`: enables internal logging
* `rust-analyzer.trace.cargo-watch`: enables cargo-watch logging


## Emacs

Prerequisites:

`emacs-lsp`, `dash` and `ht` packages.

Installation:

* add
[ra-emacs-lsp.el](https://github.com/rust-analyzer/rust-analyzer/blob/69ee5c9c5ef212f7911028c9ddf581559e6565c3/editors/emacs/ra-emacs-lsp.el)
to load path and require it in `init.el`
* run `lsp` in a rust buffer
* (Optionally) bind commands like `rust-analyzer-join-lines` or `rust-analyzer-extend-selection` to keys, and enable `rust-analyzer-inlay-hints-mode` to get inline type hints


## Vim and NeoVim

* Install coc.nvim by following the instructions at [coc.nvim]
  - You will need nodejs installed.
  - You may want to include some of the sample vim configurations [from here][coc-vim-conf]
  - Note that if you use a plugin manager other than `vim-plug`, you may need to manually
    checkout the `release` branch wherever your plugin manager cloned it. Otherwise you will
    get errors about a missing javascript file.
* Add rust analyzer using: [coc.nvim wiki][coc-wiki]
  - Use `:CocConfig` in command mode to edit the config file.

```jsonc
 "languageserver": {
  "rust": {
    "command": "ra_lsp_server",
    "filetypes": ["rust"],
    "rootPatterns": ["Cargo.toml"]
  }
 }
```

For those not familiar with js, the whole file should be enclosed in `{` and `}`, with all of your config options in between. So for example, if rust-analyzer was your only language server, you could do the following:

```jsonc
{
 "languageserver": {
  "rust": {
    "command": "ra_lsp_server",
    "filetypes": ["rust"],
    "rootPatterns": ["Cargo.toml"]
  }
 }
}
```

[coc.nvim]: https://github.com/neoclide/coc.nvim
[coc-wiki]: https://github.com/neoclide/coc.nvim/wiki/Language-servers#rust
[coc-vim-conf]: https://github.com/neoclide/coc.nvim/#example-vim-configuration


## Sublime Text 3

Prequisites:

`LSP` package.

Installation:

* Invoke the command palette with <kbd>Ctrl+Shift+P</kbd>
* Type `LSP Settings` to open the LSP preferences editor
* Add the following LSP client definition to your settings:

```json
"rust-analyzer": {
    "command": ["rustup", "run", "stable", "ra_lsp_server"],
    "languageId": "rust",
    "scopes": ["source.rust"],
    "syntaxes": [
        "Packages/Rust/Rust.sublime-syntax",
        "Packages/Rust Enhanced/RustEnhanced.sublime-syntax"
    ]
}
```

* You can now invoke the command palette and type LSP enable to locally/globally enable the rust-analyzer LSP (type LSP enable, then choose either locally or globally, then select rust-analyzer)
