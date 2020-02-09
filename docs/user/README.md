[github-releases]: https://github.com/rust-analyzer/rust-analyzer/releases

The main interface to rust-analyzer is the
[LSP](https://microsoft.github.io/language-server-protocol/) implementation. To
install lsp server, you have three options:

* **Preferred and default:** install the plugin/extension for your IDE and it will ask your permission to automatically download the latest lsp server for you from [GitHub releases][github-releases]. (See docs to find out whether this is implemented for your editor below).
* Manually download prebuilt binaries from [GitHub releases][github-releases]
    * `ra_lsp_server-linux` for Linux
    * `ra_lsp_server-mac` for Mac
    * `ra_lsp_server-windows.exe` for Windows
* Clone the repository and build from sources
```bash
$ git clone git@github.com:rust-analyzer/rust-analyzer && cd rust-analyzer
$ cargo xtask install --server # or cargo install --path ./crates/ra_lsp_server
```

This way you will get a binary named `ra_lsp_server` (with os suffix for prebuilt binaries)
which you should be able to use with any LSP-compatible editor.

We make use of custom extensions to LSP, so special client-side support is required to take full
advantage of rust-analyzer. This repository contains support code for VS Code.

Rust Analyzer needs sources of rust standard library to work, so
you might also need to execute

```
$ rustup component add rust-src
```

See [./features.md](./features.md) document for a list of features that are available.

## VS Code

### Prerequisites

You will need the most recent version of VS Code: we don't try to
maintain compatibility with older versions yet.

In order to build the VS Code plugin from sources, you need to have node.js and npm with
a minimum version of 12 installed. Please refer to
[node.js and npm documentation](https://nodejs.org) for installation instructions.

### Installation from prebuilt binaries

We ship prebuilt binaries for Linux, Mac and Windows via
[GitHub releases][github-releases].
In order to use them you need to install the client VSCode extension.

Publishing to VS Code marketplace is currently WIP. Thus, you need to manually download
`rust-analyzer-0.1.0.vsix` file from latest [GitHub release][github-releases].

After you downloaded the `.vsix` file you can install it from the terminal

```
$ code --install-extension rust-analyzer-0.1.0.vsix
```

Or open VS Code, press <kbd>Ctrl+Shift+P</kbd>, and search for the following command:

<img width="500px" alt="Install from VSIX command" src="https://user-images.githubusercontent.com/36276403/74108225-c0c11d80-4b80-11ea-9b2a-0a43f09e29af.png">

Press <kbd>Enter</kbd> and go to `rust-analyzer-0.1.0.vsix` file through the file explorer.

Then open some Rust project and you should
see an info message pop-up.

<img height="140px" src="https://user-images.githubusercontent.com/36276403/74103174-a40df100-4b52-11ea-81f4-372c70797924.png" alt="Download now message"/>


Click `Download now`, wait until the progress is 100% and you are ready to go.

For updates you need to remove installed binary
```
rm -rf ${HOME}/.config/Code/User/globalStorage/matklad.rust-analyzer
```

`"Download latest language server"` command for VSCode and automatic updates detection is currently WIP.


### Installation from sources

The experimental VS Code plugin can be built and installed by executing the
following commands:

```
$ git clone https://github.com/rust-analyzer/rust-analyzer.git --depth 1
$ cd rust-analyzer
$ cargo xtask install
```

After that you need to amend your `settings.json` file to explicitly specify the
path to `ra_lsp_server` that you've just built.
```json
{
    "rust-analyzer.raLspServerPath": "ra_lsp_server"
}
```
This should work on all platforms, otherwise if installed `ra_lsp_server` is not available through your `$PATH` then see how to configure it [here](#setting-up-the-PATH-variable).


The automatic installation is expected to *just work* for common cases, if it
doesn't, report bugs!

**Note** [#1831](https://github.com/rust-analyzer/rust-analyzer/issues/1831): If you are using the popular
[Vim emulation plugin](https://github.com/VSCodeVim/Vim), you will likely
need to turn off the `rust-analyzer.enableEnhancedTyping` setting.
(// TODO: This configuration is no longer available, enhanced typing shoud be disabled via removing Enter key binding, [see this issue](https://github.com/rust-analyzer/rust-analyzer/issues/3051))

If you have an unusual setup (for example, `code` is not in the `PATH`), you
should adapt these manual installation instructions:

```
$ git clone https://github.com/rust-analyzer/rust-analyzer.git --depth 1
$ cd rust-analyzer
$ cargo install --path ./crates/ra_lsp_server/ --force --locked
$ cd ./editors/code
$ npm install
$ npm run package
$ code --install-extension ./rust-analyzer-0.1.0.vsix
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
   the `rust-analyzer-0.1.0.vsix` file.
6. Restart Visual Studio Code and re-establish the connection to the remote host.

In case of errors please make sure that `~/.cargo/bin` is in your `PATH` on the remote
host.

### Settings

* `rust-analyzer.highlightingOn`: enables experimental syntax highlighting.
  Colors can be configured via `editor.tokenColorCustomizations`.
  As an example, [Pale Fire](https://github.com/matklad/pale-fire/) color scheme tweaks rust colors.
* `rust-analyzer.enableEnhancedTyping`: by default, rust-analyzer intercepts the
  `Enter` key to make it easier to continue comments. Note that it may conflict with VIM emulation plugin.
* `rust-analyzer.raLspServerPath`: path to `ra_lsp_server` executable, when absent or `null` defaults to prebuilt binary path
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
   ```jsonc
   {
       // Show diagnostics produced by rust-analyzer itself.
       "lsp.diagnostics": true,
       // Automatically insert `()` and `<>` when completing functions and types.
       "completion.insertion.add-call-parenthesis": true,
       // Enable completions like `.if`, `.match`, etc.
       "completion.enable-postfix": true,
       // Show notification when workspace is fully loaded
       "notifications.workspace-loaded": true,
       // Show error when no Cargo.toml was found
       "notifications.cargo-toml-not-found": true,
   }
   ```


## Emacs

* install recent version of `emacs-lsp` package by following the instructions [here][emacs-lsp]
* set `lsp-rust-server` to `'rust-analyzer`
* run `lsp` in a Rust buffer
* (Optionally) bind commands like `lsp-rust-analyzer-join-lines`, `lsp-extend-selection` and `lsp-rust-analyzer-expand-macro` to keys

[emacs-lsp]: https://github.com/emacs-lsp/lsp-mode


## Vim and NeoVim (coc-rust-analyzer)

* Install coc.nvim by following the instructions at [coc.nvim][] (nodejs required)
* Run `:CocInstall coc-rust-analyzer` to install [coc-rust-analyzer], this extension implements _most_ of the features supported in the VSCode extension:
  - same configurations as VSCode extension, `rust-analyzer.raLspServerPath`, `rust-analyzer.enableCargoWatchOnStartup` etc.
  - same commands too, `rust-analyzer.analyzerStatus`, `rust-analyzer.startCargoWatch` etc.
  - highlighting and inlay_hints are not implemented yet

[coc.nvim]: https://github.com/neoclide/coc.nvim
[coc-rust-analyzer]: https://github.com/fannheyward/coc-rust-analyzer

## Vim and NeoVim (LanguageClient-neovim)

* Install LanguageClient-neovim by following the instructions [here][lang-client-neovim]
  - The github project wiki has extra tips on configuration

* Configure by adding this to your vim/neovim config file (replacing the existing rust specific line if it exists):

```vim
let g:LanguageClient_serverCommands = {
\ 'rust': ['ra_lsp_server'],
\ }
```

[lang-client-neovim]: https://github.com/autozimu/LanguageClient-neovim

## NeoVim (nvim-lsp)

NeoVim 0.5 (not yet released) has built in language server support. For a quick start configuration
of rust-analyzer, use [neovim/nvim-lsp](https://github.com/neovim/nvim-lsp#rust_analyzer).
Once `neovim/nvim-lsp` is installed, use `lua require'nvim_lsp'.rust_analyzer.setup({})` in your `init.vim`.


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


### Setting up the `PATH` variable <!-- Update links to this header when changing it! -->

On Unix systems, `rustup` adds `~/.cargo/bin` to `PATH` by modifying the shell's
startup file. Depending on your configuration, your Desktop Environment might not
actually load it. If you find that `rust-analyzer` only runs when starting the
editor from the terminal, you will have to set up your `PATH` variable manually.

There are a couple of ways to do that:

- for Code, set `rust-analyzer.raLspServerPath` to `~/.cargo/bin` (the `~` is
  automatically resolved by the extension)
- copy the binary to a location that is already in `PATH`, e.g. `/usr/local/bin`
- on Linux, use PAM to configure the `PATH` variable, by e.g. putting
  `PATH DEFAULT=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:@{HOME}/.cargo/bin:@{HOME}/.local/bin`
  in your `~/.pam_environment` file; note that this might interfere with other
  defaults set by the system administrator via `/etc/environment`.
