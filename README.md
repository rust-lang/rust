# rust-semverver
[![Build Status](https://travis-ci.org/ibabushkin/rust-semverver.svg?branch=master)](https://travis-ci.org/ibabushkin/rust-semverver)

This repository is hosting a proof-of-concept implementation of an automatic tool checking
rust library crates for semantic versioning adherence, developed during the Google Summer
of Code 2017. The goal is to provide an automated command akin to `cargo clippy` that
analyzes the current crate's source code for changes compared to the most recent version
on `crates.io`.

## Background
The approach taken is to compile both versions of the crate to `rlib`s and to link them as
dependencies of a third, empty, crate. Then, a custom compiler driver is run on the
resulting crate and all necessary analysis is performed in that context.

More information on the inner workings will be provided soon.

## Installation
The tool is implemented as a cargo plugin. As of now, it can be obtained from this git
repository and compiled from source, provided you have a recent Rust nightly installed:

```sh
$ rustup update nightly
$ rustup default nightly
$ git clone https://github.com/ibabushkin/rust-semverver
$ cd rust-semverver
$ cargo install
```

At this point, the current development version can be invoked using `cargo semver` in any
directory your project resides in. If you prefer not to install to `~/.cargo/bin`, you can
invoke it like so after building with a regular `cargo build`:

```sh
PATH=/path/to/repo/target/debug:$PATH cargo semver <args>
```

If you have built using `cargo build --release` instead, change the path to point to the
`release` subdirectory of the `target` directory.

## Usage
Invoking `cargo semver -h` gives you the latest help message, which outlines how to use
the cargo plugin:

```sh
$ cargo semver -h
usage: cargo semver [options] [-- cargo options]

Options:
    -h, --help          print this message and exit
    -V, --version       print version information and exit
    -d, --debug         print command to debug and exit
    -s, --stable-path PATH
                        use local path as stable/old crate
    -c, --current-path PATH
                        use local path as current/new crate
    -S, --stable-pkg SPEC
                        use a name-version string as stable/old crate
    -C, --current-pkg SPEC
                        use a name-version string as current/new crate
```

## Functionality
The guideline used to implement semver compatibility is the [API evolution
RFC](https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md), which
applies the principles of semantic versioning to the Rust language. According to the RFC,
most changes are already recognized correctly, even though trait- and inherent
implementations are not yet handled, and some type checks behave incorrectly.

At the time of writing, the following types of changes are recognized and classified
correctly:

* items moving from `pub` to non-`pub`
* items changing their kind, i.e. from a `struct` to an `enum`
* additions and removals of region parameters to and from an item's declaration
* additions and removals of (possibly defaulted) type parameters to and from an item's
  declaration
* additions of new and removals of old enum variants
* additions of new and removals of old enum variant- or struct fields
* changes from tuple structs or variants to struct variants and vice-versa
* changes to a function or method's constness
* additions and removals of a self-parameter to and from methods
* addition and removal of (posslibly defaulted) trait items
* changes to the unsafety of a trait
* type changes of all toplevel items

Yet, the results presented to the user are merely an approximation of the required
versioning policy, especially at such an early stage of development.
