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

When building code that contains one or more directives of the form `extern mod P`,
rustpkg automatically searches for packages named `P` in the `RUST_PATH` (as described above).
It builds those dependencies if necessary.
Thus, when using rustpkg,
there is no need for `-L` flags to tell the linker where to find libraries for external crates.

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
A package ID can also specify a version, like:
`github.com/mozilla/rust#0.3`.
In this case, `rustpkg` will check that the repository `github.com/mozilla/rust` has a tag named `0.3`,
and report an error otherwise.
A package ID can also specify a particular revision of a repository, like:
`github.com/mozilla/rust#release-0.7`.
When the refspec (portion of the package ID after the `#`) can't be parsed as a decimal number,
rustpkg passes the refspec along to the version control system without interpreting it.
rustpkg also interprets any dependencies on such a package ID literally
(as opposed to versions, where a newer version satisfies a dependency on an older version).
Thus, `github.com/mozilla/rust#5c4cd30f80` is also a valid package ID,
since git can deduce that 5c4cd30f80 refers to a revision of the desired repository.

## Source files

rustpkg searches for four different fixed filenames in order to determine the crates to build:

* `main.rs`: Assumed to be a main entry point for building an executable.
* `lib.rs`: Assumed to be a library crate.
* `test.rs`: Assumed to contain tests declared with the `#[test]` attribute.
* `bench.rs`: Assumed to contain benchmarks declared with the `#[bench]` attribute.

## Versions

`rustpkg` packages do not need to declare their versions with an attribute inside one of the source files,
because `rustpkg` infers it from the version control system.
When building a package that is in a `git` repository,
`rustpkg` assumes that the most recent tag specifies the current version.
When building a package that is not under version control,
or that has no tags, `rustpkg` assumes the intended version is 0.1.

> **Note:** A future version of rustpkg will support semantic versions.
> Also, a future version will add the option to specify a version with a metadata
> attribute like `#[link(vers = "3.1415")]` inside the crate module,
> though this attribute will never be mandatory.

# Dependencies

rustpkg infers dependencies from `extern mod` directives.
Thus, there should be no need to pass a `-L` flag to rustpkg to tell it where to find a library.
(In the future, it will also be possible to write an `extern mod` directive referring to a remote package.)

# Custom build scripts

A file called `pkg.rs` at the root level in a workspace is called a *package script*.
If a package script exists, rustpkg executes it to build the package
rather than inferring crates as described previously.

Inside `pkg.rs`, it's possible to call back into rustpkg to finish up the build.
`rustpkg::api` contains functions to build, install, or clean libraries and executables
in the way rustpkg normally would without custom build logic.

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
