# rust-semverver

[![Travis Build Status](https://travis-ci.org/rust-dev-tools/rust-semverver.svg?branch=master)](https://travis-ci.org/rust-dev-tools/rust-semverver)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/qktm3ndv6cnbj01m?svg=true)](https://ci.appveyor.com/project/ibabushkin/rust-semverver)
[![Current Version](https://meritbadge.herokuapp.com/semverver)](https://crates.io/crates/semverver)

`rust-semverver` is a tool to check semver-compliance in Rust library crates. The core of
the tool has been developed as a student project during the Google Summer of Code 2017.

Details on the work done during GSoC 2017 can be found
[here](https://github.com/rust-dev-tools/rust-semverver/blob/master/doc/gsoc.md).

## Background

The approach taken is to compile both versions of the crate to `rlib`s and to link them as
dependencies of a third, empty, dummy crate. Then, a custom compiler driver is run on the
said dummy and all necessary analysis is performed in that context, where type information
and other resources are available.

More information on the inner workings of the tool can be found
[here](https://github.com/rust-dev-tools/rust-semverver/blob/master/doc/impl_notes.md).

## Installation

The tool is implemented as a cargo plugin. As of now, it can be obtained from this git
repository and compiled from source or installed from
[crates.io](https://crates.io/crates/semverver). Keep in mind that only the newest version
of the nighly toolchain is supported at any given time.

If you are already using Rust nightly and have successfully installed tools like
`cargo add` and `cargo clippy`, just do:

```sh
$ rustup update nightly
$ rustup component add rustc-dev --toolchain nightly
$ cargo +nightly install semverver
```

You'd also need `cmake` for some dependencies, and a few common libraries (if you hit
build failures because of missing system-wide dependencies, please open an issue, so they
can be added here).

You can also install the newest version of the tool from git:

```sh
$ rustup update nightly
$ rustup component add rustc-dev --toolchain nightly
$ cargo +nightly install --git https://github.com/rust-dev-tools/rust-semverver
```

<details>

<summary>
  Manual installation and more details
</summary>

```sh
# using rustup is recommended
$ rustup update nightly
$ rustup default nightly

$ git clone https://github.com/rust-dev-tools/rust-semverver
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

## Usage

By default, running `cargo semver` in directory with a Cargo project will try to compare
the local version the one last published on crates.io, and display warnings or errors for
all changes found.

Invoking `cargo semver -h` gives you the latest help message, which outlines how to use
the cargo plugin:

```sh
$ cargo semver -h
usage: cargo semver [options]

Options:
    -h, --help          print this message and exit
    -V, --version       print version information and exit
    -e, --explain       print detailed error explanations
    -q, --quiet         surpress regular cargo output, print only important
                        messages
        --show-public   print the public types in the current crate given by
                        -c or -C and exit
    -d, --debug         print command to debug and exit
    -a, --api-guidelines
                        report only changes that are breaking according to the
                        API-guidelines
        --features FEATURES
                        Space-separated list of features to activate
        --all-features  Activate all available features
        --no-default-features
                        Do not activate the `default` feature
        --compact       Only output the suggested version on stdout for
                        further processing
    -j, --json          Output a JSON-formatted description of all collected
                        data on stdout.
    -s, --stable-path PATH
                        use local path as stable/old crate
    -c, --current-path PATH
                        use local path as current/new crate
    -S, --stable-pkg NAME:VERSION
                        use a `name:version` string as stable/old crate
    -C, --current-pkg NAME:VERSION
                        use a `name:version` string as current/new crate
        --target <TRIPLE>
                        Build for the target triple
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

### JSON output

By passing the `-j` flag, all output on standard out is formatted as a machine-readable
JSON blob. This can be useful for integration with other tools, and always generates all
possible output (ignoring other output-related flags). The output format is defined as
follows:

The top level object contains the keys `old_version`, `new_version` and `changes`. The
former two hold a version number in the format `major.minor.patch`, the latter an object
describing changes between the crate versions, which contains two arrays in the keys
`path_changes` and `changes`.

The `path_changes` array contains objects describing item additions and removals, which
have the following keys:

* `name`: The name of the item.
* `def_span`: An object describing the location of the item in one of the crates.
* `additions`: An array of spans that describe locations where the item has been added.
* `removals`: An array of spans that describe locations where the item has been removed.

An example object might look like this:

```json
{
  "name": "NFT_META_CGROUP",
  "def_span": {
    "file": "/path/to/libc-0.2.48/src/unix/notbsd/linux/other/mod.rs",
    "line_lo": 776,
    "line_hi": 776,
    "col_lo": 0,
    "col_hi": 40
  },
  "additions": [
    {
      "file": "/path/to/libc-0.2.48/src/lib.rs",
      "line_lo": 195,
      "line_hi": 195,
      "col_lo": 16,
      "col_hi": 23
    }
  ],
  "removals": []
}
```


The `changes` array contains objects describing all other changes, which have the
following keys:

* `name`: The name of the item
* `max_category`: the most severe change category for this item, as a string.
  * Possible values are `Patch`, `NonBreaking`, `TechnicallyBreaking`, and `Breaking`.
* `new_span`: an object describing the location of the item in the new crate (see example).
* `changes`: an array of 2-element sequences containing an error message and an optional
  sub-span (`null` if none is present)

An example object might look like this:

```json
{
  "name": "<new::util::enumerate::Enumerate<T> as new::prelude::Stream>",
  "max_category": "TechnicallyBreaking",
  "new_span": {
    "file": "/path/to/tokio-0.1.17/src/util/enumerate.rs",
    "line_lo": 46,
    "line_hi": 63,
    "col_lo": 0,
    "col_hi": 1
  },
  "changes": [
    [
      "trait impl generalized or newly added",
      null
    ]
  ]
}
```

For reference, all objects describing spans have the same keys:

* `file`: A file name.
* `line_lo`: The line the span starts on.
* `line_hi`: The line the span ends on.
* `col_lo`: The column the span starts on.
* `col_hi`: The column the span ends on.

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
* correct handling of "sealed" traits
* changes to the unsafety of a trait
* type changes of all toplevel items, as well as associated items in inherent impls and
  trait definitions
* additions and removals of inherent impls or methods contained therein
* additions and removals of trait impls

Keep in mind however that the results presented to the user are merely an approximation of
the required versioning policy.

## Contributing

Please see
[CONTRIBUTING.md](https://github.com/rust-dev-tools/rust-semverver/blob/master/CONTRIBUTING.md).

## License

`rust-semverver` is distributed under the terms of the 3-clause BSD license.

See LICENSE for details.
