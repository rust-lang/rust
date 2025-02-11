# Installation

In theory, one should be able to just install the [`rust-analyzer`
binary](#rust-analyzer-language-server-binary) and have it automatically
work with any editor. We are not there yet, so some editor specific
setup is required.

Additionally, rust-analyzer needs the sources of the standard library.
If the source code is not present, rust-analyzer will attempt to install
it automatically.

To add the sources manually, run the following command:

    $ rustup component add rust-src

## Toolchain

Only the latest stable standard library source is officially supported
for use with rust-analyzer. If you are using an older toolchain or have
an override set, rust-analyzer may fail to understand the Rust source.
You will either need to update your toolchain or use an older version of
rust-analyzer that is compatible with your toolchain.

If you are using an override in your project, you can still force
rust-analyzer to use the stable toolchain via the environment variable
`RUSTUP_TOOLCHAIN`. For example, with VS Code or coc-rust-analyzer:

    { "rust-analyzer.server.extraEnv": { "RUSTUP_TOOLCHAIN": "stable" } }

## VS Code

This is the best supported editor at the moment. The rust-analyzer
plugin for VS Code is maintained [in
tree](https://github.com/rust-lang/rust-analyzer/tree/master/editors/code).

You can install the latest release of the plugin from [the
marketplace](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer).

Note that the plugin may cause conflicts with the [previous official
Rust
plugin](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust).
The latter is no longer maintained and should be uninstalled.

The server binary is stored in the extension install directory, which
starts with `rust-lang.rust-analyzer-` and is located under:

-   Linux: `~/.vscode/extensions`

-   Linux (Remote, such as WSL): `~/.vscode-server/extensions`

-   macOS: `~/.vscode/extensions`

-   Windows: `%USERPROFILE%\.vscode\extensions`

As an exception, on NixOS, the extension makes a copy of the server and
stores it under
`~/.config/Code/User/globalStorage/rust-lang.rust-analyzer`.

Note that we only support the two most recent versions of VS Code.

### Updates

The extension will be updated automatically as new versions become
available. It will ask your permission to download the matching language
server version binary if needed.

#### Nightly

We ship nightly releases for VS Code. To help us out by testing the
newest code, you can enable pre-release versions in the Code extension
page.

### Manual installation

Alternatively, download a VSIX corresponding to your platform from the
[releases](https://github.com/rust-lang/rust-analyzer/releases) page.

Install the extension with the `Extensions: Install from VSIX` command
within VS Code, or from the command line via:

    $ code --install-extension /path/to/rust-analyzer.vsix

If you are running an unsupported platform, you can install
`rust-analyzer-no-server.vsix` and compile or obtain a server binary.
Copy the server anywhere, then add the path to your settings.json, for
example:

    { "rust-analyzer.server.path": "~/.local/bin/rust-analyzer-linux" }

### Building From Source

Both the server and the Code plugin can be installed from source:

    $ git clone https://github.com/rust-lang/rust-analyzer.git && cd rust-analyzer
    $ cargo xtask install

You’ll need Cargo, nodejs (matching a supported version of VS Code) and
npm for this.

Note that installing via `xtask install` does not work for VS Code
Remote, instead you’ll need to install the `.vsix` manually.

If you’re not using Code, you can compile and install only the LSP
server:

    $ cargo xtask install --server

Make sure that `.cargo/bin` is in `$PATH` and precedes paths where
`rust-analyzer` may also be installed. Specifically, `rustup` includes a
proxy called `rust-analyzer`, which can cause problems if you’re
planning to use a source build or even a downloaded binary.

## rust-analyzer Language Server Binary

Other editors generally require the `rust-analyzer` binary to be in
`$PATH`. You can download pre-built binaries from the
[releases](https://github.com/rust-lang/rust-analyzer/releases) page.
You will need to uncompress and rename the binary for your platform,
e.g. from `rust-analyzer-aarch64-apple-darwin.gz` on Mac OS to
`rust-analyzer`, make it executable, then move it into a directory in
your `$PATH`.

On Linux to install the `rust-analyzer` binary into `~/.local/bin`,
these commands should work:

    $ mkdir -p ~/.local/bin
    $ curl -L https://github.com/rust-lang/rust-analyzer/releases/latest/download/rust-analyzer-x86_64-unknown-linux-gnu.gz | gunzip -c - > ~/.local/bin/rust-analyzer
    $ chmod +x ~/.local/bin/rust-analyzer

Make sure that `~/.local/bin` is listed in the `$PATH` variable and use
the appropriate URL if you’re not on a `x86-64` system.

You don’t have to use `~/.local/bin`, any other path like `~/.cargo/bin`
or `/usr/local/bin` will work just as well.

Alternatively, you can install it from source using the command below.
You’ll need the latest stable version of the Rust toolchain.

    $ git clone https://github.com/rust-lang/rust-analyzer.git && cd rust-analyzer
    $ cargo xtask install --server

If your editor can’t find the binary even though the binary is on your
`$PATH`, the likely explanation is that it doesn’t see the same `$PATH`
as the shell, see [this
issue](https://github.com/rust-lang/rust-analyzer/issues/1811). On Unix,
running the editor from a shell or changing the `.desktop` file to set
the environment should help.

### rustup

`rust-analyzer` is available in `rustup`:

    $ rustup component add rust-analyzer

### Arch Linux

The `rust-analyzer` binary can be installed from the repos or AUR (Arch
User Repository):

-   [`rust-analyzer`](https://www.archlinux.org/packages/extra/x86_64/rust-analyzer/)
    (built from latest tagged source)

-   [`rust-analyzer-git`](https://aur.archlinux.org/packages/rust-analyzer-git)
    (latest Git version)

Install it with pacman, for example:

    $ pacman -S rust-analyzer

### Gentoo Linux

`rust-analyzer` is installed when the `rust-analyzer` use flag is set for dev-lang/rust or dev-lang/rust-bin. You also need to set the `rust-src` use flag.

### macOS

The `rust-analyzer` binary can be installed via
[Homebrew](https://brew.sh/).

    $ brew install rust-analyzer

### Windows

It is recommended to install the latest Microsoft Visual C++ Redistributable prior to installation.
Download links can be found
[here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## VS Code or VSCodium in Flatpak

Setting up `rust-analyzer` with a Flatpak version of Code is not trivial
because of the Flatpak sandbox. While the sandbox can be disabled for
some directories, `/usr/bin` will always be mounted under
`/run/host/usr/bin`. This prevents access to the system’s C compiler, a
system-wide installation of Rust, or any other libraries you might want
to link to. Some compilers and libraries can be acquired as Flatpak
SDKs, such as `org.freedesktop.Sdk.Extension.rust-stable` or
`org.freedesktop.Sdk.Extension.llvm15`.

If you use a Flatpak SDK for Rust, it must be in your `PATH`:

 * install the SDK extensions with `flatpak install org.freedesktop.Sdk.Extension.{llvm15,rust-stable}//23.08`
 * enable SDK extensions in the editor with the environment variable `FLATPAK_ENABLE_SDK_EXT=llvm15,rust-stable` (this can be done using flatseal or `flatpak override`)

If you want to use Flatpak in combination with `rustup`, the following
steps might help:

-   both Rust and `rustup` have to be installed using
    <https://rustup.rs>. Distro packages *will not* work.

-   you need to launch Code, open a terminal and run `echo $PATH`

-   using
    [Flatseal](https://flathub.org/apps/details/com.github.tchx84.Flatseal),
    you must add an environment variable called `PATH`. Set its value to
    the output from above, appending `:~/.cargo/bin`, where `~` is the
    path to your home directory. You must replace `~`, as it won’t be
    expanded otherwise.

-   while Flatseal is open, you must enable access to "All user files"

A C compiler should already be available via `org.freedesktop.Sdk`. Any
other tools or libraries you will need to acquire from Flatpak.

## Emacs

Prerequisites: You have installed the [`rust-analyzer`
binary](#rust-analyzer-language-server-binary).

To use `rust-analyzer`, you need to install and enable one of the two
popular LSP client implementations for Emacs,
[Eglot](https://github.com/joaotavora/eglot) or [LSP
Mode](https://github.com/emacs-lsp/lsp-mode). Both enable
`rust-analyzer` by default in rust buffers if it is available.

### Eglot

Eglot is the more minimalistic and lightweight LSP client for Emacs,
integrates well with existing Emacs functionality and is built into
Emacs starting from release 29.

After installing Eglot, e.g. via `M-x package-install` (not needed from
Emacs 29), you can enable it via the `M-x eglot` command or load it
automatically in `rust-mode` via

    (add-hook 'rust-mode-hook 'eglot-ensure)

To enable clippy, you will need to configure the initialization options
to pass the `check.command` setting.

    (add-to-list 'eglot-server-programs
                 '((rust-ts-mode rust-mode) .
                   ("rust-analyzer" :initializationOptions (:check (:command "clippy")))))

For more detailed instructions and options see the [Eglot
manual](https://joaotavora.github.io/eglot) (also available from Emacs
via `M-x info`) and the [Eglot
readme](https://github.com/joaotavora/eglot/blob/master/README.md).

Eglot does not support the rust-analyzer extensions to the
language-server protocol and does not aim to do so in the future. The
[eglot-x](https://github.com/nemethf/eglot-x#rust-analyzer-extensions)
package adds experimental support for those LSP extensions.

### LSP Mode

LSP-mode is the original LSP-client for emacs. Compared to Eglot it has
a larger codebase and supports more features, like LSP protocol
extensions. With extension packages like [LSP
UI](https://github.com/emacs-lsp/lsp-mode) it offers a lot of visual
eyecandy. Further it integrates well with [DAP
mode](https://github.com/emacs-lsp/dap-mode) for support of the Debug
Adapter Protocol.

You can install LSP-mode via `M-x package-install` and then run it via
the `M-x lsp` command or load it automatically in rust buffers with

    (add-hook 'rust-mode-hook 'lsp-deferred)

For more information on how to set up LSP mode and its extension package
see the instructions in the [LSP mode
manual](https://emacs-lsp.github.io/lsp-mode/page/installation). Also
see the [rust-analyzer
section](https://emacs-lsp.github.io/lsp-mode/page/lsp-rust-analyzer/)
for `rust-analyzer` specific options and commands, which you can
optionally bind to keys.

Note the excellent
[guide](https://robert.kra.hn/posts/2021-02-07_rust-with-emacs/) from
[@rksm](https://github.com/rksm) on how to set-up Emacs for Rust
development with LSP mode and several other packages.

## Vim/Neovim

Prerequisites: You have installed the [`rust-analyzer`
binary](#rust-analyzer-language-server-binary). Not needed if the
extension can install/update it on its own, coc-rust-analyzer is one
example.

There are several LSP client implementations for Vim or Neovim:

### coc-rust-analyzer

1.  Install coc.nvim by following the instructions at
    [coc.nvim](https://github.com/neoclide/coc.nvim) (Node.js required)

2.  Run `:CocInstall coc-rust-analyzer` to install
    [coc-rust-analyzer](https://github.com/fannheyward/coc-rust-analyzer),
    this extension implements *most* of the features supported in the
    VSCode extension:

    -   automatically install and upgrade stable/nightly releases

    -   same configurations as VSCode extension,
        `rust-analyzer.server.path`, `rust-analyzer.cargo.features` etc.

    -   same commands too, `rust-analyzer.analyzerStatus`,
        `rust-analyzer.ssr` etc.

    -   inlay hints for variables and method chaining, *Neovim Only*

Note: for code actions, use `coc-codeaction-cursor` and
`coc-codeaction-selected`; `coc-codeaction` and `coc-codeaction-line`
are unlikely to be useful.

### LanguageClient-neovim

1.  Install LanguageClient-neovim by following the instructions
    [here](https://github.com/autozimu/LanguageClient-neovim)

    -   The GitHub project wiki has extra tips on configuration

2.  Configure by adding this to your Vim/Neovim config file (replacing
    the existing Rust-specific line if it exists):

        let g:LanguageClient_serverCommands = {
        \ 'rust': ['rust-analyzer'],
        \ }

### YouCompleteMe

Install YouCompleteMe by following the instructions
[here](https://github.com/ycm-core/YouCompleteMe#installation).

rust-analyzer is the default in ycm, it should work out of the box.

### ALE

To use the LSP server in [ale](https://github.com/dense-analysis/ale):

    let g:ale_linters = {'rust': ['analyzer']}

### nvim-lsp

Neovim 0.5 has built-in language server support. For a quick start
configuration of rust-analyzer, use
[neovim/nvim-lspconfig](https://github.com/neovim/nvim-lspconfig#rust_analyzer).
Once `neovim/nvim-lspconfig` is installed, use
`lua require'lspconfig'.rust_analyzer.setup({})` in your `init.vim`.

You can also pass LSP settings to the server:

    lua << EOF
    local lspconfig = require'lspconfig'

    local on_attach = function(client)
        require'completion'.on_attach(client)
    end

    lspconfig.rust_analyzer.setup({
        on_attach = on_attach,
        settings = {
            ["rust-analyzer"] = {
                imports = {
                    granularity = {
                        group = "module",
                    },
                    prefix = "self",
                },
                cargo = {
                    buildScripts = {
                        enable = true,
                    },
                },
                procMacro = {
                    enable = true
                },
            }
        }
    })
    EOF

If you're running Neovim 0.10 or later, you can enable inlay hints via `on_attach`:

```vim
lspconfig.rust_analyzer.setup({
    on_attach = function(client, bufnr)
        vim.lsp.inlay_hint.enable(true, { bufnr = bufnr })
    end
})
```

Note that the hints are only visible after `rust-analyzer` has finished loading **and** you have to
edit the file to trigger a re-render.

See <https://sharksforarms.dev/posts/neovim-rust/> for more tips on
getting started.

Check out <https://github.com/mrcjkb/rustaceanvim> for a batteries
included rust-analyzer setup for Neovim.

### vim-lsp

vim-lsp is installed by following [the plugin
instructions](https://github.com/prabirshrestha/vim-lsp). It can be as
simple as adding this line to your `.vimrc`:

    Plug 'prabirshrestha/vim-lsp'

Next you need to register the `rust-analyzer` binary. If it is avim.lspvailable
in `$PATH`, you may want to add this to your `.vimrc`:

    if executable('rust-analyzer')
      au User lsp_setup call lsp#register_server({
            \   'name': 'Rust Language Server',
            \   'cmd': {server_info->['rust-analyzer']},
            \   'whitelist': ['rust'],
            \ })
    endif

There is no dedicated UI for the server configuration, so you would need
to send any options as a value of the `initialization_options` field, as
described in the [Configuration](#configuration) section. Here is an
example of how to enable the proc-macro support:

    if executable('rust-analyzer')
      au User lsp_setup call lsp#register_server({
            \   'name': 'Rust Language Server',
            \   'cmd': {server_info->['rust-analyzer']},
            \   'whitelist': ['rust'],
            \   'initialization_options': {
            \     'cargo': {
            \       'buildScripts': {
            \         'enable': v:true,
            \       },
            \     },
            \     'procMacro': {
            \       'enable': v:true,
            \     },
            \   },
            \ })
    endif

## Sublime Text

### Sublime Text 4:

-   Follow the instructions in
    [LSP-rust-analyzer](https://github.com/sublimelsp/LSP-rust-analyzer).

Install
[LSP-file-watcher-chokidar](https://packagecontrol.io/packages/LSP-file-watcher-chokidar)
to enable file watching (`workspace/didChangeWatchedFiles`).

### Sublime Text 3:

-   Install the [`rust-analyzer`
    binary](#rust-analyzer-language-server-binary).

-   Install the [LSP package](https://packagecontrol.io/packages/LSP).

-   From the command palette, run `LSP: Enable Language Server Globally`
    and select `rust-analyzer`.

If it worked, you should see "rust-analyzer, Line X, Column Y" on the
left side of the status bar, and after waiting a bit, functionalities
like tooltips on hovering over variables should become available.

If you get an error saying `No such file or directory: 'rust-analyzer'`,
see the [`rust-analyzer` binary](#rust-analyzer-language-server-binary)
section on installing the language server binary.

## GNOME Builder

GNOME Builder 3.37.1 and newer has native `rust-analyzer` support. If
the LSP binary is not available, GNOME Builder can install it when
opening a Rust file.

## Eclipse IDE

Support for Rust development in the Eclipse IDE is provided by [Eclipse
Corrosion](https://github.com/eclipse/corrosion). If available in PATH
or in some standard location, `rust-analyzer` is detected and powers
editing of Rust files without further configuration. If `rust-analyzer`
is not detected, Corrosion will prompt you for configuration of your
Rust toolchain and language server with a link to the *Window &gt;
Preferences &gt; Rust* preference page; from here a button allows to
download and configure `rust-analyzer`, but you can also reference
another installation. You’ll need to close and reopen all .rs and Cargo
files, or to restart the IDE, for this change to take effect.

## Kate Text Editor

Support for the language server protocol is built into Kate through the
LSP plugin, which is included by default. It is preconfigured to use
rust-analyzer for Rust sources since Kate 21.12.

To change rust-analyzer config options, start from the following example
and put it into Kate’s "User Server Settings" tab (located under the LSP
Client settings):

    {
        "servers": {
            "rust": {
                "initializationOptions": {
                    "cachePriming": {
                        "enable": false
                    },
                    "check": {
                        "allTargets": false
                    },
                    "checkOnSave": false
                }
            }
        }
    }

Then click on apply, and restart the LSP server for your rust project.

## juCi++

[juCi++](https://gitlab.com/cppit/jucipp) has built-in support for the
language server protocol, and since version 1.7.0 offers installation of
both Rust and rust-analyzer when opening a Rust file.

## Kakoune

[Kakoune](https://kakoune.org/) supports LSP with the help of
[`kak-lsp`](https://github.com/kak-lsp/kak-lsp). Follow the
[instructions](https://github.com/kak-lsp/kak-lsp#installation) to
install `kak-lsp`. To configure `kak-lsp`, refer to the [configuration
section](https://github.com/kak-lsp/kak-lsp#configuring-kak-lsp) which
is basically about copying the [configuration
file](https://github.com/kak-lsp/kak-lsp/blob/master/kak-lsp.toml) in
the right place (latest versions should use `rust-analyzer` by default).

Finally, you need to configure Kakoune to talk to `kak-lsp` (see [Usage
section](https://github.com/kak-lsp/kak-lsp#usage)). A basic
configuration will only get you LSP but you can also activate inlay
diagnostics and auto-formatting on save. The following might help you
get all of this.

    eval %sh{kak-lsp --kakoune -s $kak_session}  # Not needed if you load it with plug.kak.
    hook global WinSetOption filetype=rust %{
        # Enable LSP
        lsp-enable-window

        # Auto-formatting on save
        hook window BufWritePre .* lsp-formatting-sync

        # Configure inlay hints (only on save)
        hook window -group rust-inlay-hints BufWritePost .* rust-analyzer-inlay-hints
        hook -once -always window WinSetOption filetype=.* %{
            remove-hooks window rust-inlay-hints
        }
    }

## Helix

[Helix](https://docs.helix-editor.com/) supports LSP by default.
However, it won’t install `rust-analyzer` automatically. You can follow
instructions for installing [`rust-analyzer`
binary](#rust-analyzer-language-server-binary).

## Visual Studio 2022

There are multiple rust-analyzer extensions for Visual Studio 2022 on
Windows:

### rust-analyzer.vs

(License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International)

[Visual Studio
Marketplace](https://marketplace.visualstudio.com/items?itemName=kitamstudios.RustAnalyzer)

[GitHub](https://github.com/kitamstudios/rust-analyzer/)

Support for Rust development in the Visual Studio IDE is enabled by the
[rust-analyzer](https://marketplace.visualstudio.com/items?itemName=kitamstudios.RustAnalyzer)
package. Either click on the download link or install from IDE’s
extension manager. For now [Visual Studio
2022](https://visualstudio.microsoft.com/downloads/) is required. All
editions are supported viz. Community, Professional & Enterprise. The
package aims to provide 0-friction installation and therefore comes
loaded with most things required including rust-analyzer binary. If
anything it needs is missing, appropriate errors / warnings will guide
the user. E.g. cargo.exe needs to be in path and the package will tell
you as much. This package is under rapid active development. So if you
encounter any issues please file it at
[rust-analyzer.vs](https://github.com/kitamstudios/rust-analyzer/).

### VS\_RustAnalyzer

(License: GPL)

[Visual Studio
Marketplace](https://marketplace.visualstudio.com/items?itemName=cchharris.vsrustanalyzer)

[GitHub](https://github.com/cchharris/VS-RustAnalyzer)

### SourceGear Rust

(License: closed source)

[Visual Studio
Marketplace](https://marketplace.visualstudio.com/items?itemName=SourceGear.SourceGearRust)

[GitHub (docs, issues,
discussions)](https://github.com/sourcegear/rust-vs-extension)

-   Free (no-cost)

-   Supports all editions of Visual Studio 2022 on Windows: Community,
    Professional, or Enterprise

## Lapce

[Lapce](https://lapce.dev/) has a Rust plugin which you can install
directly. Unfortunately, it downloads an old version of `rust-analyzer`,
but you can set the server path under Settings.

## Crates

There is a package named `ra_ap_rust_analyzer` available on
[crates.io](https://crates.io/crates/ra_ap_rust-analyzer), for someone
who wants to use it programmatically.

For more details, see [the publish
workflow](https://github.com/rust-lang/rust-analyzer/blob/master/.github/workflows/autopublish.yaml).

## Zed

[Zed](https://zed.dev) has native `rust-analyzer` support. If the LSP
binary is not available, Zed can install it when opening a Rust file.
