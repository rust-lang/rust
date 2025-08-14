# Suggested workflows

The full bootstrapping process takes quite a while. Here are some suggestions to
make your life easier.

## Installing a pre-push hook

CI will automatically fail your build if it doesn't pass `tidy`, our internal
tool for ensuring code quality. If you'd like, you can install a [Git
hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) that will
automatically run `./x test tidy` on each push, to ensure your code is up to
par. If the hook fails then run `./x test tidy --bless` and commit the changes.
If you decide later that the pre-push behavior is undesirable, you can delete
the `pre-push` file in `.git/hooks`.

A prebuilt git hook lives at [`src/etc/pre-push.sh`].  It can be copied into
your `.git/hooks` folder as `pre-push` (without the `.sh` extension!).

You can also install the hook as a step of running `./x setup`!

## Config extensions

When working on different tasks, you might need to switch between different bootstrap configurations.
Sometimes you may want to keep an old configuration for future use. But saving raw config values in
random files and manually copying and pasting them can quickly become messy, especially if you have a
long history of different configurations.

To simplify managing multiple configurations, you can create config extensions.

For example, you can create a simple config file named `cross.toml`:

```toml
[build]
build = "x86_64-unknown-linux-gnu"
host = ["i686-unknown-linux-gnu"]
target = ["i686-unknown-linux-gnu"]


[llvm]
download-ci-llvm = false

[target.x86_64-unknown-linux-gnu]
llvm-config = "/path/to/llvm-19/bin/llvm-config"
```

Then, include this in your `bootstrap.toml`:

```toml
include = ["cross.toml"]
```

You can also include extensions within extensions recursively.

**Note:** In the `include` field, the overriding logic follows a right-to-left order. For example,
in `include = ["a.toml", "b.toml"]`, extension `b.toml` overrides `a.toml`. Also, parent extensions
always overrides the inner ones.

## Configuring `rust-analyzer` for `rustc`

### Checking the "library" tree

Checking the "library" tree requires a stage1 compiler, which can be a heavy process on some computers.
For this reason, bootstrap has a flag called `--skip-std-check-if-no-download-rustc` that skips checking the
"library" tree if `rust.download-rustc` isn't available. If you want to avoid putting a heavy load on your computer
with `rust-analyzer`, you can add the `--skip-std-check-if-no-download-rustc` flag to your `./x check` command in
the `rust-analyzer` configuration.

### Project-local rust-analyzer setup

`rust-analyzer` can help you check and format your code whenever you save a
file. By default, `rust-analyzer` runs the `cargo check` and `rustfmt` commands,
but you can override these commands to use more adapted versions of these tools
when hacking on `rustc`. With custom setup, `rust-analyzer` can use `./x check`
to check the sources, and the stage 0 rustfmt to format them.

The default `rust-analyzer.check.overrideCommand` command line will check all
the crates and tools in the repository. If you are working on a specific part,
you can override the command to only check the part you are working on to save
checking time. For example, if you are working on the compiler, you can override
the command to `x check compiler --json-output` to only check the compiler part.
You can run `x check --help --verbose` to see the available parts.

Running `./x setup editor` will prompt you to create a project-local LSP config
file for one of the supported editors. You can also create the config file as a
step of running `./x setup`.

### Using a separate build directory for rust-analyzer

By default, when rust-analyzer runs a check or format command, it will share
the same build directory as manual command-line builds. This can be inconvenient
for two reasons:
- Each build will lock the build directory and force the other to wait, so it
  becomes impossible to run command-line builds while rust-analyzer is running
  commands in the background.
- There is an increased risk of one of the builds deleting previously-built
  artifacts due to conflicting compiler flags or other settings, forcing
  additional rebuilds in some cases.

To avoid these problems:
- Add `--build-dir=build/rust-analyzer` to all of the custom `x` commands in
  your editor's rust-analyzer configuration.
  (Feel free to choose a different directory name if desired.)
- Modify the `rust-analyzer.rustfmt.overrideCommand` setting so that it points
  to the copy of `rustfmt` in that other build directory.
- Modify the `rust-analyzer.procMacro.server` setting so that it points to the
  copy of `rust-analyzer-proc-macro-srv` in that other build directory.

Using separate build directories for command-line builds and rust-analyzer
requires extra disk space.

### Visual Studio Code

Selecting `vscode` in `./x setup editor` will prompt you to create a
`.vscode/settings.json` file which will configure Visual Studio code. The
recommended `rust-analyzer` settings live at
[`src/etc/rust_analyzer_settings.json`].

If running `./x check` on save is inconvenient, in VS Code you can use a [Build
Task] instead:

```JSON
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "./x check",
            "command": "./x check",
            "type": "shell",
            "problemMatcher": "$rustc",
            "presentation": { "clear": true },
            "group": { "kind": "build", "isDefault": true }
        }
    ]
}
```

[Build Task]: https://code.visualstudio.com/docs/editor/tasks


### Neovim

For Neovim users, there are a few options. The
easiest way is by using [neoconf.nvim](https://github.com/folke/neoconf.nvim/),
which allows for project-local configuration files with the native LSP. The
steps for how to use it are below. Note that they require rust-analyzer to
already be configured with Neovim. Steps for this can be [found
here](https://rust-analyzer.github.io/manual.html#nvim-lsp).

1. First install the plugin. This can be done by following the steps in the
   README.
2. Run `./x setup editor`, and select `vscode` to create a
   `.vscode/settings.json` file. `neoconf` is able to read and update
   rust-analyzer settings automatically when the project is opened when this
   file is detected.

If you're using `coc.nvim`, you can run `./x setup editor` and select `vim` to
create a `.vim/coc-settings.json`. The settings can be edited with
`:CocLocalConfig`. The recommended settings live at
[`src/etc/rust_analyzer_settings.json`].

Another way is without a plugin, and creating your own logic in your
configuration. The following code will work for any checkout of rust-lang/rust (newer than February 2025):

```lua
local function expand_config_variables(option)
    local var_placeholders = {
        ['${workspaceFolder}'] = function(_)
            return vim.lsp.buf.list_workspace_folders()[1]
        end,
    }

    if type(option) == "table" then
        local mt = getmetatable(option)
        local result = {}
        for k, v in pairs(option) do
            result[expand_config_variables(k)] = expand_config_variables(v)
        end
        return setmetatable(result, mt)
    end
    if type(option) ~= "string" then
        return option
    end
    local ret = option
    for key, fn in pairs(var_placeholders) do
        ret = ret:gsub(key, fn)
    end
    return ret
end
lspconfig.rust_analyzer.setup {
    root_dir = function()
        local default = lspconfig.rust_analyzer.config_def.default_config.root_dir()
        -- the default root detection uses the cargo workspace root.
        -- but for rust-lang/rust, the standard library is in its own workspace.
        -- use the git root instead.
        local compiler_config = vim.fs.joinpath(default, "../src/bootstrap/defaults/config.compiler.toml")
        if vim.fs.basename(default) == "library" and vim.uv.fs_stat(compiler_config) then
            return vim.fs.dirname(default)
        end
        return default
    end,
    on_init = function(client)
        local path = client.workspace_folders[1].name
        local config = vim.fs.joinpath(path, "src/etc/rust_analyzer_zed.json")
        if vim.uv.fs_stat(config) then
            -- load rust-lang/rust settings
            local file = io.open(config)
            local json = vim.json.decode(file:read("*a"))
            client.config.settings["rust-analyzer"] = expand_config_variables(json.lsp["rust-analyzer"].initialization_options)
            client.notify("workspace/didChangeConfiguration", { settings = client.config.settings })
        end
        return true
    end
}
```

If you would like to use the build task that is described above, you may either
make your own command in your config, or you can install a plugin such as
[overseer.nvim](https://github.com/stevearc/overseer.nvim) that can [read
VSCode's `task.json`
files](https://github.com/stevearc/overseer.nvim/blob/master/doc/guides.md#vs-code-tasks),
and follow the same instructions as above.

### Emacs

Emacs provides support for rust-analyzer with project-local configuration
through [Eglot](https://www.gnu.org/software/emacs/manual/html_node/eglot/).
Steps for setting up Eglot with rust-analyzer can be [found
here](https://rust-analyzer.github.io/manual.html#eglot).
Having set up Emacs & Eglot for Rust development in general, you can run
`./x setup editor` and select `emacs`, which will prompt you to create
`.dir-locals.el` with the recommended configuration for Eglot.
The recommended settings live at [`src/etc/rust_analyzer_eglot.el`].
For more information on project-specific Eglot configuration, consult [the
manual](https://www.gnu.org/software/emacs/manual/html_node/eglot/Project_002dspecific-configuration.html).

### Helix

Helix comes with built-in LSP and rust-analyzer support.
It can be configured through `languages.toml`, as described
[here](https://docs.helix-editor.com/languages.html).
You can run `./x setup editor` and select `helix`, which will prompt you to
create `languages.toml` with the recommended configuration for Helix. The
recommended settings live at [`src/etc/rust_analyzer_helix.toml`].

### Zed

Zed comes with built-in LSP and rust-analyzer support.
It can be configured through `.zed/settings.json`, as described
[here](https://zed.dev/docs/configuring-languages). Selecting `zed`
in `./x setup editor` will prompt you to create a `.zed/settings.json`
file which will configure Zed with the recommended configuration. The
recommended `rust-analyzer` settings live
at [`src/etc/rust_analyzer_zed.json`].

## Check, check, and check again

When doing simple refactoring, it can be useful to run `./x check`
continuously. If you set up `rust-analyzer` as described above, this will be
done for you every time you save a file. Here you are just checking that the
compiler can **build**, but often that is all you need (e.g., when renaming a
method). You can then run `./x build` when you actually need to run tests.

In fact, it is sometimes useful to put off tests even when you are not 100% sure
the code will work. You can then keep building up refactoring commits and only
run the tests at some later time. You can then use `git bisect` to track down
**precisely** which commit caused the problem. A nice side-effect of this style
is that you are left with a fairly fine-grained set of commits at the end, all
of which build and pass tests. This often helps reviewing.

## Configuring `rustup` to use nightly

Some parts of the bootstrap process uses pinned, nightly versions of tools like
rustfmt. To make things like `cargo fmt` work correctly in your repo, run

```console
cd <path to rustc repo>
rustup override set nightly
```

after [installing a nightly toolchain] with `rustup`. Don't forget to do this
for all directories you have [setup a worktree for]. You may need to use the
pinned nightly version from `src/stage0`, but often the normal `nightly` channel
will work.

**Note** see [the section on vscode] for how to configure it with this real
rustfmt `x` uses, and [the section on rustup] for how to setup `rustup`
toolchain for your bootstrapped compiler

**Note** This does _not_ allow you to build `rustc` with cargo directly. You
still have to use `x` to work on the compiler or standard library, this just
lets you use `cargo fmt`.

[installing a nightly toolchain]: https://rust-lang.github.io/rustup/concepts/channels.html?highlight=nightl#working-with-nightly-rust
[setup a worktree for]: ./suggested.md#working-on-multiple-branches-at-the-same-time
[the section on vscode]: suggested.md#configuring-rust-analyzer-for-rustc
[the section on rustup]: how-to-build-and-run.md?highlight=rustup#creating-a-rustup-toolchain

## Faster Builds with CI-rustc  

If you are not working on the compiler, you often don't need to build the compiler tree.
For example, you can skip building the compiler and only build the `library` tree or the
tools under `src/tools`. To achieve that, you have to enable this by setting the `download-rustc`
option in your configuration. This tells bootstrap to use the latest nightly compiler for `stage > 0`
steps, meaning it will have two precompiled compilers: stage0 compiler and `download-rustc` compiler
for `stage > 0` steps. This way, it will never need to build the in-tree compiler. As a result, your
build time will be significantly reduced by not building the in-tree compiler.

## Using incremental compilation

You can further enable the `--incremental` flag to save additional time in
subsequent rebuilds:

```bash
./x test tests/ui --incremental --test-args issue-1234
```

If you don't want to include the flag with every command, you can enable it in
the `bootstrap.toml`:

```toml
[rust]
incremental = true
```

Note that incremental compilation will use more disk space than usual. If disk
space is a concern for you, you might want to check the size of the `build`
directory from time to time.

## Fine-tuning optimizations

Setting `optimize = false` makes the compiler too slow for tests. However, to
improve the test cycle, you can disable optimizations selectively only for the
crates you'll have to rebuild
([source](https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/incremental.20compilation.20question/near/202712165)).
For example, when working on `rustc_mir_build`, the `rustc_mir_build` and
`rustc_driver` crates take the most time to incrementally rebuild. You could
therefore set the following in the root `Cargo.toml`:

```toml
[profile.release.package.rustc_mir_build]
opt-level = 0
[profile.release.package.rustc_driver]
opt-level = 0
```

## Working on multiple branches at the same time

Working on multiple branches in parallel can be a little annoying, since
building the compiler on one branch will cause the old build and the incremental
compilation cache to be overwritten. One solution would be to have multiple
clones of the repository, but that would mean storing the Git metadata multiple
times, and having to update each clone individually.

Fortunately, Git has a better solution called [worktrees]. This lets you create
multiple "working trees", which all share the same Git database. Moreover,
because all of the worktrees share the same object database, if you update a
branch (e.g. master) in any of them, you can use the new commits from any of the
worktrees. One caveat, though, is that submodules do not get shared. They will
still be cloned multiple times.

[worktrees]: https://git-scm.com/docs/git-worktree

Given you are inside the root directory for your Rust repository, you can create
a "linked working tree" in a new "rust2" directory by running the following
command:

```bash
git worktree add ../rust2
```

Creating a new worktree for a new branch based on `master` looks like:

```bash
git worktree add -b my-feature ../rust2 master
```

You can then use that rust2 folder as a separate workspace for modifying and
building `rustc`!

## Working with nix

Several nix configurations are defined in `src/tools/nix-dev-shell`.

If you're using direnv, you can create a symbol link to `src/tools/nix-dev-shell/envrc-flake` or `src/tools/nix-dev-shell/envrc-shell`

```bash
ln -s ./src/tools/nix-dev-shell/envrc-flake ./.envrc # Use flake
```
or
```bash
ln -s ./src/tools/nix-dev-shell/envrc-shell ./.envrc # Use nix-shell
```

### Note

Note that when using nix on a not-NixOS distribution, it may be necessary to set
**`patch-binaries-for-nix = true` in `bootstrap.toml`**. Bootstrap tries to detect
whether it's running in nix and enable patching automatically, but this
detection can have false negatives.

You can also use your nix shell to manage `bootstrap.toml`:

```nix
let
  config = pkgs.writeText "rustc-config" ''
    # Your bootstrap.toml content goes here
  ''
pkgs.mkShell {
  /* ... */
  # This environment variable tells bootstrap where our bootstrap.toml is.
  RUST_BOOTSTRAP_CONFIG = config;
}
```

## Shell Completions

If you use Bash, Zsh, Fish or PowerShell, you can find automatically-generated shell
completion scripts for `x.py` in
[`src/etc/completions`](https://github.com/rust-lang/rust/tree/master/src/etc/completions).

You can use `source ./src/etc/completions/x.py.<extension>` to load completions
for your shell of choice, or `& .\src\etc\completions\x.py.ps1` for PowerShell.
Adding this to your shell's startup script (e.g. `.bashrc`) will automatically
load this completion.

[`src/etc/rust_analyzer_settings.json`]: https://github.com/rust-lang/rust/blob/master/src/etc/rust_analyzer_settings.json
[`src/etc/rust_analyzer_eglot.el`]: https://github.com/rust-lang/rust/blob/master/src/etc/rust_analyzer_eglot.el
[`src/etc/rust_analyzer_helix.toml`]: https://github.com/rust-lang/rust/blob/master/src/etc/rust_analyzer_helix.toml
[`src/etc/rust_analyzer_zed.json`]: https://github.com/rust-lang/rust/blob/master/src/etc/rust_analyzer_zed.json
[`src/etc/pre-push.sh`]: https://github.com/rust-lang/rust/blob/master/src/etc/pre-push.sh
