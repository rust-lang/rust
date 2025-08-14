# Other Editors

rust-analyzer works with any editor that supports the [Language Server
Protocol](https://microsoft.github.io/language-server-protocol/).

This page assumes that you have already [installed the rust-analyzer
binary](./rust_analyzer_binary.html).

<!-- toc -->

## Emacs

To use `rust-analyzer`, you need to install and enable one of the two
popular LSP client implementations for Emacs,
[Eglot](https://github.com/joaotavora/eglot) or [LSP
Mode](https://github.com/emacs-lsp/lsp-mode). Both enable
`rust-analyzer` by default in Rust buffers if it is available.

### Eglot

Eglot is the more minimalistic and lightweight LSP client for Emacs,
integrates well with existing Emacs functionality and is built into
Emacs starting from release 29.

After installing Eglot, e.g. via `M-x package-install` (not needed from
Emacs 29), you can enable it via the `M-x eglot` command or load it
automatically in `rust-mode` via

```
(add-hook 'rust-mode-hook 'eglot-ensure)
```

To enable clippy, you will need to configure the initialization options
to pass the `check.command` setting.

```
(add-to-list 'eglot-server-programs
             '((rust-ts-mode rust-mode) .
               ("rust-analyzer" :initializationOptions (:check (:command "clippy")))))
```

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

```
(add-hook 'rust-mode-hook 'lsp-deferred)
```

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

Note: coc-rust-analyzer is capable of installing or updating the
rust-analyzer binary on its own.

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

```lua
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
```

If you're running Neovim 0.10 or later, you can enable inlay hints via `on_attach`:

```lua
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

-   Install the [LSP package](https://packagecontrol.io/packages/LSP).

-   From the command palette, run `LSP: Enable Language Server Globally`
    and select `rust-analyzer`.

If it worked, you should see "rust-analyzer, Line X, Column Y" on the
left side of the status bar, and after waiting a bit, functionalities
like tooltips on hovering over variables should become available.

If you get an error saying `No such file or directory: 'rust-analyzer'`,
see the [rust-analyzer binary installation](./rust_analyzer_binary.html) section.

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

```json
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
```

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
instructions for [installing the rust-analyzer
binary](./rust_analyzer_binary.html).

## Visual Studio 2022

There are multiple rust-analyzer extensions for Visual Studio 2022 on
Windows:

### VS RustAnalyzer

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

## Zed

[Zed](https://zed.dev) has native `rust-analyzer` support. If the
rust-analyzer binary is not available, Zed can install it when opening
a Rust file.
