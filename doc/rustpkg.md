% Rustpkg Reference Manual

# Introduction

This document is the reference manual for the Rustpkg packaging and build tool for the Rust programming language.

## Disclaimer

Rustpkg is a work in progress, as is this reference manual.
If the actual behavior of rustpkg differs from the behavior described in this reference,
that reflects either an incompleteness or a bug in rustpkg.

# Package searching

rustpkg searches for packages using the `RUST_PATH` environment variable,
which is a colon-separated list (semicolon-separated on Windows) of directories.

Each directory in this list is a *workspace* for rustpkg.

`RUST_PATH` implicitly contains an entry for `./.rust` (as well as
`../.rust`, `../../.rust`,
and so on for every parent of `.` up to the filesystem root).
That means that if `RUST_PATH` is not set,
then rustpkg will still search for workspaces in `./.rust` and so on.
`RUST_PATH` also implicitly contains an entry for the system path:
`/usr/local` or the equivalent on Windows.
This entry comes after the implicit entries for `./.rust` and so on.
Finally, the last implicit entry in `RUST_PATH` is `~/.rust`
or the equivalent on Windows.

Each workspace may contain one or more packages.

# Package structure

A valid workspace must contain each of the following subdirectories:

* 'src/': contains one subdirectory per package. Each subdirectory contains source files for a given package.

     For example, if `foo` is a workspace containing the package `bar`,
     then `foo/src/bar/main.rs` could be the `main` entry point for
     building a `bar` executable.
* 'lib/': `rustpkg install` installs libraries into a target-specific subdirectory of this directory.

     For example, on a 64-bit machine running Mac OS X,
     if `foo` is a workspace containing the package `bar`,
     rustpkg will install libraries for bar to `foo/lib/x86_64-apple-darwin/`.
     The libraries will have names of the form `foo/lib/x86_64-apple-darwin/libbar-[hash].dylib`,
     where [hash] is a hash of the package ID.
* 'bin/': `rustpkg install` installs executable binaries into a target-specific subdirectory of this directory.

     For example, on a 64-bit machine running Mac OS X,
     if `foo` is a workspace, containing the package `bar`,
     rustpkg will install executables for `bar` to
     `foo/bin/x86_64-apple-darwin/`.
     The executables will have names of the form `foo/bin/x86_64-apple-darwin/bar`.
* 'build/': `rustpkg build` stores temporary build artifacts in a target-specific subdirectory of this directory.

     For example, on a 64-bit machine running Mac OS X,
     if `foo` is a workspace containing the package `bar` and `foo/src/bar/main.rs` exists,
     then `rustpkg build` will create `foo/build/x86_64-apple-darwin/bar/main.o`.

# Package identifiers

A package identifier identifies a package uniquely.
A package can be stored in a workspace on the local file system,
or on a remote Web server, in which case the package ID resembles a URL.
For example, `github.com/mozilla/rust` is a package ID
that would refer to the git repository browsable at `http://github.com/mozilla/rust`.

## Source files

rustpkg searches for four different fixed filenames in order to determine the crates to build:

* `main.rs`: Assumed to be a main entry point for building an executable.
* `lib.rs`: Assumed to be a library crate.
* `test.rs`: Assumed to contain tests declared with the `#[test]` attribute.
* `bench.rs`: Assumed to contain benchmarks declared with the `#[bench]` attribute.

# Custom build scripts

A file called `pkg.rs` at the root level in a workspace is called a *package script*.
If a package script exists, rustpkg executes it to build the package
rather than inferring crates as described previously.

# Command reference

## build

`rustpkg build foo` searches for a package with ID `foo`
and builds it in any workspace(s) where it finds one.
Supposing such packages are found in workspaces X, Y, and Z,
the command leaves behind files in `X`'s, `Y`'s, and `Z`'s `build` directories,
but not in their `lib` or `bin` directories.

## clean

`rustpkg clean foo` deletes the contents of `foo`'s `build` directory.

## install

`rustpkg install foo` builds the libraries and/or executables that are targets for `foo`,
and then installs them either into `foo`'s `lib` and `bin` directories,
or into the `lib` and `bin` subdirectories of the first entry in `RUST_PATH`.

## test

`rustpkg test foo` builds `foo`'s `test.rs` file if necessary,
then runs the resulting test executable.
