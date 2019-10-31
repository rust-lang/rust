The main interface to rust-analyzer is the
[LSP](https://microsoft.github.io/language-server-protocol/) implementation. To
install lsp server, use `cargo xtask install --server`, which is a shorthand for `cargo
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
$ cargo xtask install
```

The automatic installation is expected to *just work* for common cases, if it
doesn't, report bugs!

If you have an unusual setup (for example, `code` is not in the `PATH`), you
should adapt these manual installation instructions:

```
$ git clone https://github.com/rust-analyzer/rust-analyzer.git --depth 1
$ cd rust-analyzer
$ cargo install --path ./crates/ra_lsp_server/ --force --locked
$ cd ./editors/code
$ npm install
$ ./node_modules/vsce/out/vsce package
$ code --install-extension ./ra-lsp-0.0.1.vsix
```

It's better to remove existing Rust plugins to avoid interference.

Beyond basic LSP features, there are some extension commands which you can
invoke via <kbd>Ctrl+Shift+P</kbd> or bind to a shortcut. See [./features.md](./features.md)
for details.

For updates, pull the latest changes from the master branch, run `cargo xtask install` again, and **restart** VS Code instance.
See [microsoft/vscode#72308](https://github.com/microsoft/vscode/issues/72308) for why a full restart is needed.

### VS Code Remote

You can also use `rust-analyzer` with the Visual Studio Code Remote extensions
(Remote SSH, Remote WSL, Remote Containers). In this case, however, you have to
manually install the `.vsix` package:

1. Build the extension on the remote host using the instructions above (ignore the
   error if `code` cannot be found in your PATH: VSCode doesn't need to be installed
   on the remote host).
2. In Visual Studio Code open a connection to the remote host.
3. Open the Extensions View (`View > Extensions`, keyboard shortcut: `Ctrl+Shift+X`).
4. From the top-right kebab menu (`···`) select `Install from VSIX...`
5. Inside the `rust-analyzer` directory find the `editors/code` subdirectory and choose
   the `ra-lsp-0.0.1.vsix` file.
6. Restart Visual Studio Code and re-establish the connection to the remote host.

In case of errors please make sure that `~/.cargo/bin` is in your `PATH` on the remote
host.

### Settings

* `rust-analyzer.highlightingOn`: enables experimental syntax highlighting
* `rust-analyzer.enableEnhancedTyping`: by default, rust-analyzer intercepts
  `Enter` key to make it easier to continue comments. Note that it may conflict with VIM emulation plugin.
* `rust-analyzer.raLspServerPath`: path to `ra_lsp_server` executable
* `rust-analyzer.enableCargoWatchOnStartup`: prompt to install & enable `cargo
  watch` for live error highlighting (note, this **does not** use rust-analyzer)
* `rust-analyzer.excludeGlobs`: a list of glob-patterns for exclusion (see globset [docs](https://docs.rs/globset) for syntax).
  Note: glob patterns are applied to all Cargo packages and a rooted at a package root.
  This is not very intuitive and a limitation of a current implementation.
* `rust-analyzer.useClientWatching`: use client provided file watching instead
  of notify watching.
* `rust-analyzer.cargo-watch.command`: `cargo-watch` command. (e.g: `clippy` will run as `cargo watch -x clippy` )
* `rust-analyzer.cargo-watch.arguments`: cargo-watch check arguments.
  (e.g: `--features="shumway,pdf"` will run as `cargo watch -x "check --features="shumway,pdf""` )
* `rust-analyzer.cargo-watch.ignore`: list of patterns for cargo-watch to ignore (will be passed as `--ignore`)
* `rust-analyzer.trace.server`: enables internal logging
* `rust-analyzer.trace.cargo-watch`: enables cargo-watch logging
* `RUST_SRC_PATH`: environment variable that overwrites the sysroot
* `rust-analyzer.featureFlags` -- a JSON object to tweak fine-grained behavior:
   ```js
   {
       // Show diagnostics produced by rust-analyzer itself.
       "lsp.diagnostics": true,
       // Automatically insert `()` and `<>` when completing functions and types.
       "completion.insertion.add-call-parenthesis": true,
       // Show notification when workspace is fully loaded
       "notifications.workspace-loaded": true,
   }
   ```


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
* Run `:CocInstall coc-rust-analyzer` to install [coc-rust-analyzer], this extension implemented _most_ of the features supported in the VSCode extension:
  - same configurations as VSCode extension, `rust-analyzer.raLspServerPath`, `rust-analyzer.enableCargoWatchOnStartup` etc.
  - same commands too, `rust-analyzer.analyzerStatus`, `rust-analyzer.startCargoWatch` etc.
  - highlighting and inlay_hints are not implemented yet

[coc.nvim]: https://github.com/neoclide/coc.nvim
[coc-vim-conf]: https://github.com/neoclide/coc.nvim/#example-vim-configuration
[coc-rust-analyzer]: https://github.com/fannheyward/coc-rust-analyzer

## Vim and NeoVim Alternative

* Install LanguageClient-neovim by following the instructions [here][lang-client-neovim]
  - No extra run-time is required as this server is written in Rust
  - The github project wiki has extra tips on configuration

* Configure by adding this to your vim/neovim config file (replacing the existing rust specific line if it exists):

```
let g:LanguageClient_serverCommands = {
\ 'rust': ['ra_lsp_server'],
\ }
```

[lang-client-neovim]: https://github.com/autozimu/LanguageClient-neovim


## Sublime Text 3

Prequisites:

`LSP` package.

Installation:

* Invoke the command palette with <kbd>Ctrl+Shift+P</kbd>
* Type `LSP Settings` to open the LSP preferences editor
* Add the following LSP client definition to your settings:

```json
"rust-analyzer": {
    "command": ["ra_lsp_server"],
    "languageId": "rust",
    "scopes": ["source.rust"],
    "syntaxes": [
        "Packages/Rust/Rust.sublime-syntax",
        "Packages/Rust Enhanced/RustEnhanced.sublime-syntax"
    ],
    "initializationOptions": {
      "featureFlags": {
      }
    },
}
```

* You can now invoke the command palette and type LSP enable to locally/globally enable the rust-analyzer LSP (type LSP enable, then choose either locally or globally, then select rust-analyzer)

* Note that `ra_lsp_server` binary must be in `$PATH` for this to work. If it's not the case, you can specify full path to the binary, which is typically `.cargo/bin/ra_lsp_server`.
