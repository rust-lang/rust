- Feature Name: N/A
- Start Date: 2015-07-10
- RFC PR: [rust-lang/rfcs#1200](https://github.com/rust-lang/rfcs/pull/1200)
- Rust Issue: N/A

# Summary

Add a new subcommand to Cargo, `install`, which will install `[[bin]]`-based
packages onto the local system in a Cargo-specific directory.

# Motivation

There has [almost always been a desire][cargo-37] to be able to install Cargo
packages locally, but it's been somewhat unclear over time what the precise
meaning of this is. Now that we have crates.io and lots of experience with
Cargo, however, the niche that `cargo install` would fill is much clearer.

[cargo-37]: https://github.com/rust-lang/cargo/issues/37

Fundamentally, however, Cargo is a ubiquitous tool among the Rust community and
implementing `cargo install` would facilitate sharing Rust code among its
developers.  Simple tasks like installing a new cargo subcommand, installing an
editor plugin, etc, would be just a `cargo install` away. Cargo can manage
dependencies and versions itself to make the process as seamless as possible.

Put another way, enabling easily sharing code is one of Cargo's fundamental
design goals, and expanding into binaries is simply an extension of Cargo's core
functionality.

# Detailed design

The following new subcommand will be added to Cargo:

```
Install a crate onto the local system

Installing new crates:
    cargo install [options]
    cargo install [options] [-p CRATE | --package CRATE] [--vers VERS]
    cargo install [options] --git URL [--branch BRANCH | --tag TAG | --rev SHA]
    cargo install [options] --path PATH

Managing installed crates:
    cargo install [options] --list

Options:
    -h, --help              Print this message
    -j N, --jobs N          The number of jobs to run in parallel
    --features FEATURES     Space-separated list of features to activate
    --no-default-features   Do not build the `default` feature
    --debug                 Build in debug mode instead of release mode
    --bin NAME              Only install the binary NAME
    --example EXAMPLE       Install the example EXAMPLE instead of binaries
    -p, --package CRATE     Install this crate from crates.io or select the
                            package in a repository/path to install.
    -v, --verbose           Use verbose output
    --root                  Directory to install packages into

This command manages Cargo's local set of install binary crates. Only packages
which have [[bin]] targets can be installed, and all binaries are installed into
`$HOME/.cargo/bin` by default (or `$CARGO_HOME/bin` if you change the home
directory).

There are multiple methods of installing a new crate onto the system. The
`cargo install` command with no arguments will install the current crate (as
specifed by the current directory). Otherwise the `-p`, `--package`, `--git`,
and `--path` options all specify the source from which a crate is being
installed. The `-p` and `--package` options will download crates from crates.io.

Crates from crates.io can optionally specify the version they wish to install
via the `--vers` flags, and similarly packages from git repositories can
optionally specify the branch, tag, or revision that should be installed. If a
crate has multiple binaries, the `--bin` argument can selectively install only
one of them, and if you'd rather install examples the `--example` argument can
be used as well.

The `--list` option will list all installed packages (and their versions).
```

## Installing Crates

Cargo attempts to be as flexible as possible in terms of installing crates from
various locations and specifying what should be installed. All binaries will be
stored in a **cargo-local** directory, and more details on where exactly this is
located can be found below.

Cargo will not attempt to install binaries or crates into system directories
(e.g. `/usr`) as that responsibility is intended for system package managers.

To use installed crates one just needs to add the binary path to their `PATH`
environment variable. This will be recommended when `cargo install` is run if
`PATH` does not already look like it's configured.

#### Crate Sources

The `cargo install` command will be able to install crates from any source that
Cargo already understands. For example it will start off being able to install
from crates.io, git repositories, and local paths. Like with normal
dependencies, downloads from crates.io can specify a version, git repositories
can specify branches, tags, or revisions.

#### Sources with multiple crates

Sources like git repositories and paths can have multiple crates inside them,
and Cargo needs a way to figure out which one is being installed. If there is
more than one crate in a repo (or path), then Cargo will apply the following
heuristics to select a crate, in order:

1. If the `-p` argument is specified, use that crate.
2. If only one crate has binaries, use that crate.
3. If only one crate has examples, use that crate.
4. Print an error suggesting the `-p` flag.

#### Multiple binaries in a crate

Once a crate has been selected, Cargo will by default build all binaries and
install them. This behavior can be modified with the `--bin` or `--example`
flags to configure what's installed on the local system.

#### Building a Binary

The `cargo install` command has some standard build options found on `cargo
build` and friends, but a key difference is that `--release` is the default for
installed binaries so a `--debug` flag is present to switch this back to
debug-mode. Otherwise the `--features` flag can be specified to activate various
features of the crate being installed.

The `--target` option is omitted as `cargo install` is not intended for creating
cross-compiled binaries to ship to other platforms.

#### Conflicting Crates

Cargo will not namespace the installation directory for crates, so conflicts may
arise in terms of binary names. For example if crates A and B both provide a
binary called `foo` they cannot be both installed at once. Cargo will reject
these situations and recommend that a binary is selected via `--bin` or the
conflicting crate is uninstalled.

#### Placing output artifacts

The `cargo install` command can be customized where it puts its output artifacts
to install packages in a custom location. The root directory of the installation
will be determined in a hierarchical fashion, choosing the first of the
following that is specified:

1. The `--root` argument on the command line.
2. The environment variable `CARGO_INSTALL_ROOT`.
3. The `install.root` configuration option.
4. The value of `$CARGO_HOME` (also determined in an independent and
   hierarchical fashion).

Once the root directory is found, Cargo will place all binaries in the
`$INSTALL_ROOT/bin` folder. Cargo will also reserve the right to retain some
metadata in this folder in order to keep track of what's installed and what
binaries belong to which package.

## Managing Installations

If Cargo gives access to installing packages, it should surely provide the
ability to manage what's installed! The first part of this is just discovering
what's installed, and this is provided via `cargo install --list`.

## Removing Crates

To remove an installed crate, another subcommand will be added to Cargo:

```
Remove a locally installed crate

Usage:
    cargo uninstall [options] SPEC

Options:
    -h, --help              Print this message
    --bin NAME              Only uninstall the binary NAME
    --example EXAMPLE       Only uninstall the example EXAMPLE
    -v, --verbose           Use verbose output

The argument SPEC is a package id specification (see `cargo help pkgid`) to
specify which crate should be uninstalled. By default all binaries are
uninstalled for a crate but the `--bin` and `--example` flags can be used to
only uninstall particular binaries.
```

Cargo won't remove the source for uninstalled crates, just the binaries that
were installed by Cargo itself.

## Non-binary artifacts

Cargo will not currently attempt to manage anything other than a binary artifact
of `cargo build`. For example the following items will not be available to
installed crates:

* Dynamic native libraries built as part of `cargo build`.
* Native assets such as images not included in the binary itself.
* The source code is not guaranteed to exist, and the binary doesn't know where
  the source code is.

Additionally, Cargo will not immediately provide the ability to configure the
installation stage of a package. There is often a desire for a "pre-install
script" which runs various house-cleaning tasks. This is left as a future
extension to Cargo.

# Drawbacks

Beyond the standard "this is more surface area" and "this may want to
aggressively include more features initially" concerns there are no known
drawbacks at this time.

# Alternatives

### System Package Managers

The primary alternative to putting effort behind `cargo install` is to instead
put effort behind system-specific package managers. For example the line between
a system package manager and `cargo install` is a little blurry, and the
"official" way to distribute a package should in theory be through a system
package manager. This also has the upside of benefiting those outside the Rust
community as you don't have to have Cargo installed to manage a program. This
approach is not without its downsides, however:

* There are *many* system package managers, and it's unclear how much effort it
  would be for Cargo to support building packages for all of them.
* Actually preparing a package for being packaged in a system package manager
  can be quite onerous and is often associated with a high amount of overhead.
* Even once a system package is created, it must be added to an online
  repository in one form or another which is often different for each
  distribution.

All in all, even if Cargo invested effort in facilitating creation of system
packages, **the threshold for distribution a Rust program is still too high**.
If everything went according to plan it's just unfortunately inherently complex
to only distribute packages through a system package manager because of the
various requirements and how diverse they are. The `cargo install` command
provides a cross-platform, easy-to-use, if Rust-specific interface to installing
binaries.

It is expected that all major Rust projects will still invest effort into
distribution through standard package managers, and Cargo will certainly have
room to help out with this, but it doesn't obsolete the need for
`cargo install`.

### Installing Libraries

Another possibility for `cargo install` is to not only be able to install
binaries, but also libraries. The meaning of this however, is pretty nebulous
and it's not clear that it's worthwhile. For example all Cargo builds will not
have access to these libraries (as Cargo retains control over dependencies). It
may mean that normal invocations of `rustc` have access to these libraries (e.g.
for small one-off scripts), but it's not clear that this is worthwhile enough to
support installing libraries yet.

Another possible interpretation of installing libraries is that a developer is
informing Cargo that the library should be available in a pre-compiled form. If
any compile ends up using the library, then it can use the precompiled form
instead of recompiling it. This job, however, seems best left to `cargo build`
as it will automatically handle when the compiler version changes, for example.
It may also be more appropriate to add the caching layer at the `cargo build`
layer instead of `cargo install`.

# Unresolved questions

None yet
