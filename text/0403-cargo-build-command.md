- Start Date: 2014-10-30
- RFC PR: [rust-lang/rfcs#403](https://github.com/rust-lang/rfcs/pull/403)
- Rust Issue: [rust-lang/rust#18473](https://github.com/rust-lang/rust/issues/18473)

# Summary

Overhaul the `build` command internally and establish a number of conventions
around build commands to facilitate linking native code to Cargo packages.

1. Instead of having the `build` command be some form of script, it will be a
   Rust command instead
2. Establish a namespace of `foo-sys` packages which represent the native
   library `foo`. These packages will have Cargo-based dependencies between
   `*-sys` packages to express dependencies among C packages themselves.
3. Establish a set of standard environment variables for build commands which
   will instruct how `foo-sys` packages should be built in terms of dynamic or
   static linkage, as well as providing the ability to override where a package
   comes from via environment variables.

# Motivation

Building native code is normally quite a tricky business, and the original
design of Cargo was to essentially punt on this problem. Today's "solution"
involves invoking an arbitrary `build` command in a sort of pseudo-shell with a
number of predefined environment variables. This ad-hoc solution was known to be
lacking at the time of implementing with the intention of identifying major pain
points over time and revisiting the design once we had more information.

While today's "hands off approach" certainly has a number of drawbacks, one of
the upsides is that Cargo minimizes the amount of logic inside it as much as
possible. This proposal attempts to stress this point as much as possible by
providing a strong foundation on which to build robust build scripts, but not
baking all of the logic into Cargo itself.

The time has now come to revisit the design, and some of the largest pain points
that have been identified are:

1. Packages needs the ability to build differently on different platforms.
2. Projects should be able to control dynamic vs static at the top level. Note
   that the term "project" here means "top level package".
3. It should be possible to use libraries of build tool functionality. Cargo is
   indeed a package manager after all, and currently there is no way share a
   common set of build tool functionality among different Cargo packages.
4. There is very little flexibility in locating packages, be it on the system,
   in a build directory, or in a home build dir.
5. There is no way for two Rust packages to declare that they depend on the same
   native dependency.
6. There is no way for C libraries to express their dependence on other C
   libraries.
7. There is no way to encode a platform-specific dependency.

Each of these concerns can be addressed somewhat ad-hocly with a vanilla `build`
command, but Cargo can certainly provide a more comprehensive solution to these
problems.

Most of these concerns are fairly self-explanatory, but specifically (2) may
require a bit more explanation:

## Selecting linkage from the top level

Conceptually speaking, a native library is largely just a collections of
symbols. The linkage involved in creating a final product is an implementation
detail that is almost always irrelevant with respect to the symbols themselves.

When it comes to linking a native library, there are often a number of
overlapping and sometimes competing concerns:

1. Most unix-like distributions with package managers highly recommend dynamic
   linking of all dependencies. This reduces the overall size of an installation
   and allows dependencies to be updated without updating the original
   application.
2. Those who distribute binaries of an application to many platforms prefer
   static linking as much as possible. This is largely done because the actual
   set of libraries on the platforms being installed on are often unknown and
   could be quite different than those linked to. Statically linking solves
   these problems by reducing the number of dependencies for an application.
3. General developers of a package simply want a package to build at all costs.
   It's ok to take a little bit longer to build, but if it takes hours of
   googling obscure errors to figure out you needed to install `libfoo` it's
   probably not ok.
4. Some native libraries have obscure linkage requirements. For example OpenSSL
   on OSX likely wants to be linked dynamically due to the special keychain
   support, but on linux it's more ok to statically link OpenSSL if necessary.

The key point here is that the author of a library is not the one who dictates
how an application should be linked. The builder or packager of a library is the
one responsible for determining how a package should be linked.

Today this is not quite how Cargo operates, depending on what flavor of syntax
extension you may be using. One of the goals of this re-working is to enable
top-level projects to make easier decisions about how to link to libraries,
where to find linked libraries, etc.

# Detailed design

Summary:

* Add a `-l` flag to rustc
* Tweak an `include!` macro to rustc
* Add a `links` key to Cargo manifests
* Add platform-specific dependencies to Cargo manifests
* Allow pre-built libraries in the same manner as Cargo overrides
* Use Rust for build scripts
* Develop a convention of `*-sys` packages

## Modifications to `rustc`

A new flag will be added to `rustc`:

```
    -l LIBRARY          Link the generated crate(s) to the specified native
                        library LIBRARY. The name `LIBRARY` will have the format
                        `kind:name` where `kind` is one of: dylib, static,
                        framework. This corresponds to the `kind` key of the
                        `#[link]` attribute. The `name` specified is the name of
                        the native library to link. The `kind:` prefix may be
                        omitted and the `dylib` format will be assumed.
```

```
rustc -l dylib:ssl -l static:z foo.rs
```

Native libraries often have widely varying dependencies depending on what
platforms they are compiled on. Often times these dependencies aren't even
constant among one platform! The reality we sadly have to face is that the
dependencies of a native library itself are sometimes unknown until *build
time*, at which point it's too late to modify the source code of the program to
link to a library.

For this reason, the `rustc` CLI will grow the ability to link to arbitrary
libraries at build time. This is motivated by the build scripts which Cargo is
growing, but it likely useful for custom Rust compiles at large.

Note that this RFC does not propose style guidelines nor suggestions for usage
of `-l` vs `#[link]`. For Cargo it will later recommend discouraging use of
`#[link]`, but this is not generally applicable to all Rust code in existence.

## Declaration of native library dependencies

Today Cargo has very little knowledge about what dependencies are being used by
a package. By knowing the exact set of dependencies, Cargo paves a way into the
future to extend its handling of native dependencies, for example downloading
precompiled libraries. This extension allows Cargo to better handle constraint 5
above.

```toml
[package]

# This package unconditionally links to this list of native libraries
links = ["foo", "bar"]
```

The key `links` declares that the package will link to and provide the given C
libraries. Cargo will impose the restriction that the same C library *must not*
appear more than once in a dependency graph. This will prevent the same C
library from being linked multiple times to packages.

If conflicts arise from having multiple packages in a dependency graph linking
to the same C library, the C dependency should be refactored into a common
Cargo-packaged dependency.

It is illegal to define `links` without also defining `build`.

## Platform-specific dependencies

A number of native dependencies have various dependencies depending on what
platform they're building for. For example, libcurl does not depend on OpenSSL
on Windows, but it is a common dependency on unix-based systems. To this end,
Cargo will gain support for platform-specific dependencies, solving constriant 7
above:

```toml

[target.i686-pc-windows-gnu.dependencies.crypt32]
git = "https://github.com/user/crypt32-rs"

[target.i686-pc-windows-gnu.dependencies.winhttp]
path = "winhttp"
```

Here the top-level configuration key `target` will be a table whose sub-keys
are target triples. The dependencies section underneath is the same as the
top-level dependencies section in terms of functionality.

Semantically, platform specific dependencies are activated whenever Cargo is
compiling for a the exact target. Dependencies in other `$target` sections
will not be compiled.

However, when generating a lockfile, Cargo will always download all dependencies
unconditionally and perform resolution as if all packages were included. This is
done to prevent the lockfile from radically changing depending on whether the
package was last built on Linux or windows. This has the advantage of a stable
lockfile, but has the drawback that all dependencies must be downloaded, even if
they're not used.

## Pre-built libraries

A common pain point with constraints 1, 2, and cross compilation is that it's
occasionally difficult to compile a library for a particular platform. Other
times it's often useful to have a copy of a library locally which is linked
against instead of built or detected otherwise for debugging purposes (for
example). To facilitate these pain points, Cargo will support pre-built
libraries being on the system similar to how local package overrides are
available.

Normal Cargo configuration will be used to specify where a library is and how
it's supposed to be linked against:

```toml
# Each target triple has a namespace under the global `target` key and the
# `libs` key is a table for each native library.
#
# Each library can specify a number of key/value pairs where the values must be
# strings. The key/value pairs are metadata which are passed through to any
# native build command which depends on this library. The `rustc-flags` key is
# specially recognized as a set of flags to pass to `rustc` in order to link to
# this library.
[target.i686-unknown-linux-gnu.ssl]
rustc-flags = "-l static:ssl -L /home/build/root32/lib"
root = "/home/build/root32"
```

This configuration will be placed in the normal locations that `.cargo/config`
is found. The configuration will only be queried if the target triple being
built matches what's in the configuration.

## Rust build scripts

First pioneered by @tomaka in https://github.com/rust-lang/cargo/issues/610, the
`build` command will no longer be an actual command, but rather a build script
itself. This decision is motivated in solving constraints 1 and 3 above. The
major motivation for this recommendation is the realization that the only common
denominator for platforms that Cargo is running on is the fact that a Rust
compiler is available. The natural conclusion from this fact is for a build
script is to use Rust itself.

Furthermore, Cargo itself which serves quite well as a dependency manager, so by
using Rust as a build tool it will be able to manage dependencies of the build
tool itself. This will allow third-party solutions for build tools to be
developed outside of Cargo itself and shared throughout the ecosystem of
packages.

The concrete design of this will be the `build` command in the manifest being a
relative path to a file in the package:

```toml
[package]
# ...
build = "build/compile.rs"
```

This file will be considered the entry point as a "build script" and will be
built as an executable. A new top-level dependencies array, `build-dependencies`
will be added to the manifest. These dependencies will all be available to the
build script as external crates. Requiring that the build command have a
separate set of dependencies solves a number of constraints:

* When cross-compiling, the build tool as well as all of its dependencies are
  required to be built for the host architecture instead of the target
  architecture. A clear deliniation will indicate precisely what dependencies
  need to be built for the host architecture.
* Common packages, such as one to build `cmake`-based dependencies, can develop
  conventions around filesystem hierarchy formats to require minimum
  configuration to build extra code while being easily identified as having
  extra support code.

This RFC does not propose a convention of what to name the build script files.

Unlike `links`, it will be legal to specify `build` without specifying `links`.
This is motivated by the code generation case study below.

### Inputs

Cargo will provide a number of inputs to the build script to facilitate building
native code for the current package:

* The `TARGET` environment variable will contain the target triple that the
  native code needs to be built for. This will be passed unconditionally.
* The `NUM_JOBS` environment variable will indicate the number of parallel jobs
  that the script itself should execute (if relevant).
* The `CARGO_MANIFEST_DIR` environment variables will be the directory of the
  manifest of the package being built. Note that this is not the directory of
  the package whose build command is being run.
* The `OPT_LEVEL` environment variable will contain the requested optimization
  level of code being built. This will be in the range 0-2. Note that this
  variable is the same for all build commands.
* The `PROFILE` environment variable will contain the currently active Cargo
  profile being built. Note that this variable is the same for all build
  commands.
* The `DEBUG` environment variable will contain `true` or `false` depending on
  whether the current profile specified that it should be debugged or not. Note
  that this variable is the same for all build commands.
* The `OUT_DIR` environment variables contains the location in which all output
  should be placed. This should be considered a scratch area for compilations of
  any bundled items.
* The `CARGO_FEATURE_<foo>` environment variable will be present if the feature
  `foo` is enabled. for the package being compiled.
* The `DEP_<foo>_<key>` environment variables will contain metadata about the
  native dependencies for the current package. As the output section below will
  indicate, each compilation of a native library can generate a set of output
  metadata which will be passed through to dependencies. The only dependencies
  available (`foo`) will be those in `links` for immediate dependencies of the
  package being built. Note that each metadata `key` will be uppercased and `-`
  characters transformed to `_` for the name of the environment variable.
* If `links` is not present, then the command is unconditionally run with 0
  command line arguments, otherwise:
* The libraries that are requested via `links` are passed as command line
  arguments. The pre-built libraries in `links` (detailed above) will be
  filtered out and not passed to the build command. If there are no libraries to
  build (they're all pre-built), the build command will not be invoked.

### Outputs

The responsibility of the build script is to ensure that all requested native
libraries are available for the crate to compile. The conceptual output of the
build script will be metadata on stdout explaining how the compilation
went and whether it succeeded.

An example output of a build command would be:

```
cargo:rustc-flags=-l static:foo -L /path/to/foo
cargo:root=/path/to/foo
cargo:libdir=/path/to/foo/lib
cargo:include=/path/to/foo/include
```

Each line that begins with `cargo:` is interpreted as a line of metadata for
Cargo to store. The remaining part of the line is of the form `key=value` (like
environment variables).

This output is similar to the pre-built libraries section above in that most
key/value pairs are opaque metadata except for the special `rustc-flags` key.
The `rustc-flags` key indicates to Cargo necessary flags needed to link the
libraries specified.

For `rustc-flags` specifically, Cargo will propagate all `-L` flags transitively
to all dependencies, and `-l` flags to the package being built. All metadata
will only be passed to immediate dependants. Note that this is recommending that
`#[link]` is discouraged as it is not the source code's responsibility to
dictate linkage.

If the build script exits with a nonzero exit code, then Cargo will consider it
to have failed and will abort compilation.

### Input/Output rationale

In general one of the purposes of a custom build command is to dynamically
determine the necessary dependencies for a library. These dependencies may have
been discovered through `pkg-config`, built locally, or even downloaded from a
remote. This set can often change, and is the impetus for the `rustc-flags`
metadata key. This key indicates what libraries should be linked (and how) along
with where to find the libraries.

The remaining metadata flags are not as useful to `rustc` itself, but are quite
useful to interdependencies among native packages themselves. For example
libssh2 depends on OpenSSL on linux, which means it needs to find the
corresponding libraries and header files. The metadata keys serve as a vector
through which this information can be transmitted. The maintainer of the
`openssl-sys` package (described below) would have a build script responsible
for generating this sort of metadata so consumer packages can use it to build C
libraries themselves.

## A set of `*-sys` packages

This section will discuss a *convention* by which Cargo packages providing
native dependencies will be named, it is not proposed to have Cargo enforce this
convention via any means. These conventions are proposed to address constraints
5 and 6 above.

Common C dependencies will be refactored into a package named `foo-sys` where
`foo` is the name of the C library that `foo-sys` will provide and link to.
There are two key motivations behind this convention:

* Each `foo-sys` package will declare its own dependencies on other `foo-sys`
  based packages
* Dependencies on native libraries expressed through Cargo will be subject to
  version management, version locking, and deduplication as usual.

Each `foo-sys` package is responsible for providing the following:

* Declarations of all symbols in a library. Essentially each `foo-sys` library
  is *only* a header file in terms of Rust-related code.
* Ensuring that the native library `foo` is linked to the `foo-sys` crate. This
  guarantees that all exposed symbols are indeed linked into the crate.

Dependencies making use of `*-sys` packages will not expose `extern` blocks
themselves, but rather use the symbols exposed in the `foo-sys` package
directly. Additionally, packages using `*-sys` packages should not declare a
`#[link]` directive to link to the native library as it's already linked to the
`*-sys` package.

## Phasing strategy

The modifications to the `build` command are breaking changes to Cargo. To ease
the transition, the build comand will be join'd to the root path of a crate, and
if the file exists and ends with `.rs`, it will be compiled as describe above.
Otherwise a warning will be printed and the fallback behavior will be
executed.

The purpose of this is to help most build scripts today continue to work (but
not necessarily all), and pave the way forward to implement the newer
integration.

## Case study: Cargo

Cargo has a surprisingly complex set of C dependencies, and this proposal has
created an [example repository][example] for what the configuration of Cargo
would look like with respect to its set of C dependencies.

[example]: https://github.com/alexcrichton/complicated-linkage-example

## Case study: generated code

As the release of Rust 1.0 comes closer, the use of complier plugins has become
increasingly worrying over time. It is likely that plugins will not be available
by default in the stable and beta release channels of Rust. Many core Cargo
packages in the ecosystem today, such as gl-rs and iron, depend on plugins
to build. Others, like rust-http, are already using compile-time code generation
with a build script (which this RFC will attempt to standardize on).

When taking a closer look at these crates' dependence on plugins it's discovered
that the primary use case is generating Rust code at compile time. For gl-rs,
this is done to bind a platform-specific and evolving API, and for rust-http
this is done to make code more readable and easier to understand. In general
generating code at compile time is quite a useful ability for other applications
such as bindgen (C bindings), dom bindings (used in Servo), etc.

Cargo's and Rust's support for compile-time generated code is quite lacking
today, and overhauling the `build` command provides a nice opportunity to
rethink this sort of functionality.

With this motivation, this RFC proposes tweaking the `include!` macro to enable
it to be suitable for the purpose of including generated code:

```rust
include!(concat!(env!("OUT_DIR"), "/generated.rs"));
```

Today this does not compile as the argument to `include!` must be a string
literal. This RFC proposes tweaking the semantics of the `include!` macro to
expand locally before testing for a string literal. This is similar to the
behavior of the `format_args!` macro today.

Using this, Cargo crates will have `OUT_DIR` present for compilations, and any
generated Rust code can be generated by the `build` command and placed into
`OUT_DIR`. The `include!` macro would then be used to include the contents of
the code inside of the appropriate module.

## Case study: controlling linkage

One of the motivations for this RFC and redesign of the `build` command is to
making linkage controls more explicit to Cargo itself rather than hardcoding
particular linkages in source code. As proposed, however, this RFC does not bake
any sort of dynamic-vs-static knowledge into Cargo itself.

This design area is intentionally left untouched by Cargo in order to reduce the
number of moving parts and also in an effort to simplify build commands as much
as possible. There are, however, a number of methods to control how libraries
are linked:

1. First and foremost is the ability to override libraries via Cargo
   configuration. Overridden native libraries are specified manually and
   override whatever the "default" would have been otherwise.
2. Delegation to arbitrary code running in build scripts allow the possibility
   of specification through other means such as environment variables.
3. Usage of common third-party build tools will allow for conventions about
   selecting linkage to develop over time.

Note that points 2 and 3 are intentionally vague as this RFC does not have a
specific recommendation for how scripts or tooling should respect linkage. By
relying on a common set of dependencies to find native libraries it is
envisioned that the tools will grow a convention through which a linkage
preference can be specified.

For example, a possible implementation of `pkg-config` will be discussed. This
tool can be used as a first-line-defense to help locate a library on the system
as well as its dependencies. If a crate requests that `pkg-config` find the
library `foo`, then the `pkg-config` crate could inspect some environments
variables for how it operates:

* If `FOO_NO_PKG_CONFIG` is set, then pkg-config immediately returns an errors.
  This helps users who want to force pkg-config to not find a package or force
  the package to build a statically linked fallback.
* If `FOO_DYNAMIC` is set, then pkg-config will only succeed if it finds a
  dynamic version of `foo`. A similar meaning could be applied to `FOO_STATIC`.
* If `PKG_CONFIG_ALL_DYNAMIC` is set, then it will act as if the package `foo`
  is requested by be dynamic specifically (similarly for static linking).

Note that this is not a concrete design, this is just meant to be an example to
show how a common third-party tool can develop a convention for controlling
linkage not through Cargo itself.

Also note that this can mean that `cargo` itself may not succeed "by default" in
all cases, or larger projects with more flavorful configurations may want to
pursue more fine-tuned control over how libraries are linked. It is intended
that `cargo` will itself be driven with something such as a `Makefile` to
perform this configuration (be it environment or in files).

# Drawbacks

* The system proposed here for linking native code is in general somewhat
  verbose.  In theory well designed third-party Cargo crates can alleviate this
  verbosity by providing much of the boilerplate, but it's unclear to what
  extent they'll be able to alleviate it.
* None of the third-party crates with "convenient build logic" currently exist,
  and it will take time to build these solutions.
* Platform specific dependencies mean that the entire package graph must always
  be downloaded, regardless of the platform.
* In general dealing with linkage is quite complex, and the conventions/systems
  proposed here aren't exactly trivial and may be overkill for these purposes.

* As can be seen in the [example repository][verbose], platform dependencies are
  quite verbose and are difficult to work with when you actually want a negation
  instead of a positive platform to include.
* Features themselves will also likely need to be platform-specific, but this
  runs into a number of tricky situations and needs to be fleshed out.

[verbose]: https://github.com/alexcrichton/complicated-linkage-example/blob/master/curl-sys/Cargo.toml#L9-L17

# Alternatives

* It has been proposed to support the `links` manifest key in the `features`
  section as well. In the proposed scheme you would have to create an optional
  dependency representing an optional native dependency, but this may be too
  burdensome for some cases.

* The build command could instead take a script from an external package to run
  instead of a script inside of the package itself. The major drawback of this
  approach is that even the tiniest of build scripts require a full-blown
  package which needs to be uploaded to the registry and such. Due to the
  verboseness of so many packages, this was decided against.

* Cargo remains fairly "dumb" with respect to how native libraries are linked,
  and it's always a possibility that Cargo could grow more first-class support
  for dealing with the linkage of C libraries.

# Unresolved questions

None
