# Basics for hacking on Clippy

This document explains the basics for hacking on Clippy. Besides others, this
includes how to build and test Clippy. For a more in depth description on
the codebase take a look at [Adding Lints] or [Common Tools].

[Adding Lints]: https://github.com/rust-lang/rust-clippy/blob/master/doc/adding_lints.md
[Common Tools]: https://github.com/rust-lang/rust-clippy/blob/master/doc/common_tools_writing_lints.md

- [Basics for hacking on Clippy](#basics-for-hacking-on-clippy)
  - [Get the Code](#get-the-code)
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
cargo test --test dogfood
```

If the output of a [UI test] differs from the expected output, you can update the
reference file with:

```bash
cargo dev bless
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
cargo dev ra_setup
```

## PR

We follow a rustc no merge-commit policy.
See <https://rustc-dev-guide.rust-lang.org/contributing.html#opening-a-pr>.
