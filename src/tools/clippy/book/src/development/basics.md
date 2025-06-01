# Basics for hacking on Clippy

This document explains the basics for hacking on Clippy. Besides others, this
includes how to build and test Clippy. For a more in depth description on the
codebase take a look at [Adding Lints] or [Common Tools].

[Adding Lints]: adding_lints.md
[Common Tools]: common_tools_writing_lints.md

- [Basics for hacking on Clippy](#basics-for-hacking-on-clippy)
  - [Get the Code](#get-the-code)
  - [Building and Testing](#building-and-testing)
  - [`cargo dev`](#cargo-dev)
  - [lintcheck](#lintcheck)
  - [PR](#pr)
  - [Common Abbreviations](#common-abbreviations)
  - [Install from source](#install-from-source)

## Get the Code

First, make sure you have checked out the latest version of Clippy. If this is
your first time working on Clippy, create a fork of the repository and clone it
afterwards with the following command:

```bash
git clone git@github.com:<your-username>/rust-clippy
```

If you've already cloned Clippy in the past, update it to the latest version:

```bash
# If the upstream remote has not been added yet
git remote add upstream https://github.com/rust-lang/rust-clippy
# upstream has to be the remote of the rust-lang/rust-clippy repo
git fetch upstream
# make sure that you are on the master branch
git checkout master
# rebase your master branch on the upstream master
git rebase upstream/master
# push to the master branch of your fork
git push
```

## Building and Testing

You can build and test Clippy like every other Rust project:

```bash
cargo build  # builds Clippy
cargo test   # tests Clippy
```

Since Clippy's test suite is pretty big, there are some commands that only run a
subset of Clippy's tests:

```bash
# only run UI tests
cargo uitest
# only run UI tests starting with `test_`
TESTNAME="test_" cargo uitest
# only run dogfood tests
cargo dev dogfood
```

If the output of a [UI test] differs from the expected output, you can update
the reference file with:

```bash
cargo bless
```

For example, this is necessary if you fix a typo in an error message of a lint,
or if you modify a test file to add a test case.

> _Note:_ This command may update more files than you intended. In that case
> only commit the files you wanted to update.

[UI test]: https://rustc-dev-guide.rust-lang.org/tests/adding.html#ui-test-walkthrough

## `cargo dev`

Clippy has some dev tools to make working on Clippy more convenient. These tools
can be accessed through the `cargo dev` command. Available tools are listed
below. To get more information about these commands, just call them with
`--help`.

```bash
# formats the whole Clippy codebase and all tests
cargo dev fmt
# register or update lint names/groups/...
cargo dev update_lints
# create a new lint and register it
cargo dev new_lint
# deprecate a lint and attempt to remove code relating to it
cargo dev deprecate
# automatically formatting all code before each commit
cargo dev setup git-hook
# (experimental) Setup Clippy to work with IntelliJ-Rust
cargo dev setup intellij
# runs the `dogfood` tests
cargo dev dogfood
```

More about [intellij] command usage and reasons.

[intellij]: https://github.com/rust-lang/rust-clippy/blob/master/CONTRIBUTING.md#intellij-rust

## lintcheck

`cargo lintcheck` will build and run Clippy on a fixed set of crates and
generate a log of the results.  You can `git diff` the updated log against its
previous version and see what impact your lint made on a small set of crates.
If you add a new lint, please audit the resulting warnings and make sure there
are no false positives and that the suggestions are valid.

Refer to the tools [README] for more details.

[README]: https://github.com/rust-lang/rust-clippy/blob/master/lintcheck/README.md

## PR

We follow a rustc no merge-commit policy. See
<https://rustc-dev-guide.rust-lang.org/contributing.html#opening-a-pr>.

## Common Abbreviations

| Abbreviation | Meaning                                |
|--------------|----------------------------------------|
| UB           | Undefined Behavior                     |
| FP           | False Positive                         |
| FN           | False Negative                         |
| ICE          | Internal Compiler Error                |
| AST          | Abstract Syntax Tree                   |
| MIR          | Mid-Level Intermediate Representation  |
| HIR          | High-Level Intermediate Representation |
| TCX          | Type context                           |

This is a concise list of abbreviations that can come up during Clippy
development. An extensive general list can be found in the [rustc-dev-guide
glossary][glossary]. Always feel free to ask if an abbreviation or meaning is
unclear to you.

## Install from source

If you are hacking on Clippy and want to install it from source, do the
following:

From the Clippy project root, run the following command to build the Clippy
binaries and copy them into the toolchain directory. This will create a new
toolchain called `clippy` by default, see `cargo dev setup toolchain --help`
for other options.

```terminal
cargo dev setup toolchain
```

Now you may run `cargo +clippy clippy` in any project using the new toolchain.

```terminal
cd my-project
cargo +clippy clippy
```

...or `clippy-driver`

```terminal
clippy-driver +clippy <filename>
```

If you no longer need the toolchain it can be uninstalled using `rustup`:

```terminal
rustup toolchain uninstall clippy
```

> **DO NOT** install using `cargo install --path . --force` since this will
> overwrite rustup
> [proxies](https://rust-lang.github.io/rustup/concepts/proxies.html). That is,
> `~/.cargo/bin/cargo-clippy` and `~/.cargo/bin/clippy-driver` should be hard or
> soft links to `~/.cargo/bin/rustup`. You can repair these by running `rustup
> update`.

[glossary]: https://rustc-dev-guide.rust-lang.org/appendix/glossary.html
