# Basics for hacking on Clippy

This document explains the basics for hacking on Clippy. Besides others, this
includes how to set-up the development environment, how to build and how to test
Clippy. For a more in depth description on the codebase take a look at [Adding
Lints] or [Common Tools].

[Adding Lints]: https://github.com/rust-lang/rust-clippy/blob/master/doc/adding_lints.md
[Common Tools]: https://github.com/rust-lang/rust-clippy/blob/master/doc/common_tools_writing_lints.md

- [Basics for hacking on Clippy](#basics-for-hacking-on-clippy)
  - [Get the code](#get-the-code)
  - [Setup](#setup)
  - [Building and Testing](#building-and-testing)
  - [`cargo dev`](#cargo-dev)
  - [PR](#pr)

## Get the Code

First, make sure you have checked out the latest version of Clippy. If this is
your first time working on Clippy, create a fork of the repository and clone it
afterwards with the following command:

```bash
git clone git@github.com:<your-username>/rust-clippy
```

If you've already cloned Clippy in the past, update it to the latest version:

```bash
# upstream has to be the remote of the rust-lang/rust-clippy repo
git fetch upstream
# make sure that you are on the master branch
git checkout master
# rebase your master branch on the upstream master
git rebase upstream/master
# push to the master branch of your fork
git push
```

## Setup

Next we need to setup the toolchain to compile Clippy. Since Clippy heavily
relies on compiler internals it is build with the latest rustc master. To get
this toolchain, you can just use the `setup-toolchain.sh` script or use
`rustup-toolchain-install-master`:

```bash
bash setup-toolchain.sh
# OR
cargo install rustup-toolchain-install-master
# For better IDE integration also add `-c rustfmt -c rust-src` (optional)
rustup-toolchain-install-master -f -n master -c rustc-dev -c llvm-tools
rustup override set master
```

_Note:_ Sometimes you may get compiler errors when building Clippy, even if you
didn't change anything. Normally those will be fixed by a maintainer in a few hours. 

## Building and Testing

Once the `master` toolchain is installed, you can build and test Clippy like
every other Rust project:

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
cargo test --test dogfood
```

If the output of a [UI test] differs from the expected output, you can update the
reference file with:

```bash
sh tests/ui/update-all-references.sh
```

For example, this is necessary, if you fix a typo in an error message of a lint
or if you modify a test file to add a test case.

_Note:_ This command may update more files than you intended. In that case only
commit the files you wanted to update.

[UI test]: https://rustc-dev-guide.rust-lang.org/tests/adding.html#guide-to-the-ui-tests

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
# (experimental) Setup Clippy to work with rust-analyzer
cargo dev ra-setup
```

## PR

We follow a rustc no merge-commit policy.
See <https://rustc-dev-guide.rust-lang.org/contributing.html#opening-a-pr>.
