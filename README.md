# rust-semverver

[![Build
Status](https://travis-ci.org/rust-lang-nursery/rust-semverver.svg?branch=master)](https://travis-ci.org/rust-lang-nursery/rust-semverver)
[![Current Version](https://meritbadge.herokuapp.com/semverver)](https://crates.io/crates/semverver)

This repository is hosting a proof-of-concept implementation of an automatic tool checking
rust library crates for semantic versioning adherence, developed during the Google Summer
of Code 2017. The goal is to provide an automated command akin to `cargo clippy` that
analyzes the current crate's source code for changes compared to the most recent version
on `crates.io`.

Details on the work done during GSoC 2017 can be found
[here](https://github.com/rust-lang-nursery/rust-semverver/blob/master/doc/gsoc.md).

## Background

The approach taken is to compile both versions of the crate to `rlib`s and to link them as
dependencies of a third, empty, dummy crate. Then, a custom compiler driver is run on the
said dummy and all necessary analysis is performed in that context, where type information
and other resources are available.

More information on the inner workings of the tool can be found
[here](https://github.com/rust-lang-nursery/rust-semverver/blob/master/doc/impl_notes.md).

## Installation

The tool is implemented as a cargo plugin. As of now, it can be obtained from this git
repository and compiled from source.

If you are already using Rust nightly and have successfully installed tools like
`cargo add` and `cargo clippy`, just do:

```sh
$ cargo +nightly install --git https://github.com/rust-lang-nursery/rust-semverver
```

<details>

<summary>
  Manual installation and more details
</summary>

```sh
# using rustup is recommended
$ rustup update nightly
$ rustup default nightly

$ git clone https://github.com/rust-lang-nursery/rust-semverver
$ cd rust-semverver
$ cargo install
```

At this point, the current development version can be invoked using `cargo semver` in any
directory your project resides in. If you prefer not to install to `~/.cargo/bin`, you can
invoke it like so after building with a regular `cargo build`:

```sh
$ PATH=/path/to/repo/target/debug:$PATH cargo semver <args>
```

If you have built using `cargo build --release` instead, change the path to point to the
`release` subdirectory of the `target` directory.

</details>

You can also install the (possibly more stable) version of the tool from crates.io:

```sh
$ cargo install +nightly semverver
```

## Usage

By default, running `cargo semver` in directory with a Cargo project will try to compare
the local version the one last published on crates.io, and display warnings or errors for
all changes found.

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
    -S, --stable-pkg NAME:VERSION
                        use a `name:version` string as stable/old crate
    -C, --current-pkg NAME:VERSION
                        use a `name:version` string as current/new crate
```

This means that you can compare any two crates' specified versions, as long as they are
available on crates.io or present on your filesystem.

### CI setup

Assuming you use a CI provider that gives you access to cargo, you can use the following
snippet to check your build for semver compliance, and enforce that version bumps are
carried out correctly with regards to the current version of your crate on crates.io:

```sh
# install a current version of rust-semverver
cargo install semverver
# fetch the version in the manifest of your crate (adapt this to your usecase if needed)
eval "current_version=$(grep -e '^version = .*$' Cargo.toml | cut -d ' ' -f 3)"
# run the semver checks and output them for convenience
cargo semver | tee semver_out
# fail the build if necessary
(head -n 1 semver_out | grep "\-> $current_version") || (echo "versioning mismatch" && return 1)
```

Make sure you do the above with access to a nightly toolchain. Check your CI provider's
documentation on how to do that.

## Functionality

The guideline used to implement semver compatibility is the [API evolution
RFC](https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md), which
applies the principles of semantic versioning to the Rust language's semantics. According
to the RFC, most changes are already recognized correctly, even though some type checks
still behave incorrectly in edge-cases. A longterm goal is to fix this in the compiler.

At the time of writing, the following types of changes are recognized and classified
correctly:

* items moving from `pub` to non-`pub` and vice-versa
* items changing their kind, i.e. from a `struct` to an `enum`
* additions and removals of region parameters to and from an item's declaration
* additions and removals of (possibly defaulted) type parameters to and from an item's
  declaration
* changes to the variance of type and region parameters
* additions and removals of enum variants
* additions and removals of enum variant- or struct fields
* changes from tuple structs or variants to struct variants and vice-versa
* changes to a function or method's constness
* additions and removals of a self-parameter on methods
* additions and removals of (posslibly defaulted) trait items
* changes to the unsafety of a trait
* type changes of all toplevel items, as well as associated items in inherent impls and
  trait definitions
* additions and removals of inherent impls or methods contained therein
* additions and removals of trait impls

Yet, the results presented to the user are merely an approximation of the required
versioning policy, especially at such an early stage of development.
