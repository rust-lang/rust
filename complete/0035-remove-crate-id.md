- Start Date: 2014-06-24
- RFC PR: [rust-lang/rfcs#109](https://github.com/rust-lang/rfcs/pull/109)
- Rust Issue: [rust-lang/rust#14470](https://github.com/rust-lang/rust/issues/14470)

# Summary

* Remove the `crate_id` attribute and knowledge of versions from rustc.
* Add a `#[crate_name]` attribute similar to the old `#[crate_id]` attribute
* Filenames will no longer have versions, nor will symbols
* A new flag, `--extern`, will be used to override searching for external crates
* A new flag, `-C metadata=foo`, used when hashing symbols

# Motivation

The intent of CrateId and its support has become unclear over time as the
initial impetus, `rustpkg`, has faded over time. With `cargo` on the horizon,
doubts have been cast on the compiler's support for dealing with crate
versions and friends. The goal of this RFC is to simplify the compiler's
knowledge about the identity of a crate to allow cargo to do all the necessary
heavy lifting.

This new crate identification is designed to not compromise on the usability of
the compiler independent of cargo. Additionally, all use cases support today
with a CrateId should still be supported.

# Detailed design

A new `#[crate_name]` attribute will be accepted by the compiler, which is the
equivalent of the old `#[crate_id]` attribute, except without the "crate id"
support. This new attribute can have a string value describe a valid crate name.

A crate name must be a valid rust identifier with the exception of allowing the
`-` character after the first character.

```rust
#![crate_name = "foo"]
#![crate_type = "lib"]

pub fn foo() { /* ... */ }
```

## Naming library filenames

Currently, rustc creates filenames for library following this pattern:

```
lib<name>-<version>-<hash>.rlib
```

The current scheme defines `<hash>` to be the hash of the CrateId value. This
naming scheme achieves a number of goals:

* Libraries of the same name can exist next to one another if they have
  different versions.
* Libraries of the same name and version, but from different sources, can exist
  next to one another due to having different hashes.
* Rust libraries can have very privileged names such as `core` and `std` without
  worrying about polluting the global namespace of other system libraries.

One drawback of this scheme is that the output filename of the compiler is
unknown due to the `<hash>` component. One must query `rustc` itself to
determine the name of the library output.

Under this new scheme, the new output filenames by the compiler would be:

```
lib<name>.rlib
```

Note that both the `<version>` and the `<hash>` are missing by default. The
`<version>` was removed because the compiler no longer knows about the version,
and the `<hash>` was removed to make the output filename predictable.

The three original goals can still be satisfied with this simplified naming
scheme. As explained in th enext section, the compiler's "glob pattern" when
searching for a crate named `foo` will be `libfoo*.rlib`, which will help
rationalize some of these conclusions.

* Libraries of the same name can exist next to one another because they can be
  manually renamed to have extra data after the `libfoo`, such as the version.
* Libraries of the same name and version, but different source, can also exist
  by modifing what comes after `libfoo`, such as including a hash.
* Rust does not need to occupy a privileged namespace as the default rust
  installation would include hashes in all the filenames as necessary. More on
  this later.

Additionally, with a predictable filename output external tooling should be
easier to write.

## Loading crates

The goal of the crate loading phase of the compiler is to map a set of `extern
crate` statements to (dylib,rlib) pairs that are present on the filesystem. To
do this, the current system matches dependencies via the CrateId syntax:

```rust
extern crate json = "super-fast-json#0.1.0";
```

In today's compiler, this directive indicates that the a filename of the form
`libsuper-fast-json-0.1.0-<hash>.rlib` must be found to be a candidate. Further
checking happens once a candidate is found to ensure that it is indeed a rust
library.

Concerns have been raised that this key point of dependency management is where
the compiler is doing work that is not necessarily its prerogative. In a
cargo-driven world, versions are primarily managed in an external manifest, in
addition to doing other various actions such as renaming packages at compile
time.

One solution would be to add more version management to the compiler, but this
is seen as the compiler delving too far outside what it was initially tasked to
do. With this in mind, this is the new proposal for the `extern crate` syntax:

```rust
extern crate json = "super-fast-json";
```

Notably, the CrateId is removed entirely, along with the version and path
associated with it. The string value of the `extern crate` directive is still
optional (defaulting to the identifier), and the string must be a valid crate
name (as defined above).

The compiler's searching and file matching logic would be altered to only match
crates based on name. If two versions of a crate are found, the compiler will
unconditionally emit an error. It will be up to the user to move the two
libraries on the filesystem and control the `-L` flags to the compiler to enable
disambiguation.

This imples that when the compiler is searching for the crate named `foo`, it
will search all of the lookup paths for files which match the pattern
`libfoo*.{so,rlib}`. This is likely to return many false positives, but they
will be easily weeded out once the compiler realizes that there is no metadata
in the library.

This scheme is strictly less powerful than the previous, but it moves a good
deal of logic from the compiler to cargo.

### Manually specifying dependencies

Cargo is often seen as "expert mode" in its usage of the compiler. Cargo will
always have prior knowledge about what exact versions of a library will be used
for any particular dependency, as well as where the outputs are located.

If the compiler provided no support for loading crates beyond matching
filenames, it would limit many of cargo's use cases. For example, cargo could
not compile a crate with two different versions of an upstream crate.
Additionally, cargo could not substitute `libfast-json` for `libslow-json` at
compile time (assuming they have the same API).

To accomodate an "expert mode" in rustc, the compiler will grow a new command
line flag of the form:

```
--extern json=path/to/libjson
```

This directive will indicate that the library `json` can be found at
`path/to/libjson`. The file extension is not specified, and it is assume that
the rlib/dylib pair are located next to one another at this location (`libjson`
is the file stem).

This will enable cargo to drive how the compiler loads crates by manually
specifying where files are located and exactly what corresponds to what.

## Symbol mangling

Today, mangled symbols contain the version number at the end of the symbol
itself. This was originally intended to tie into Linux's ability to version
symbols, but in retrospect this is generally viewed as over-ambitious as the
support is not currently there, nor does it work on windows or OSX.

Symbols would no longer contain the version number anywhere within them. The
hash at the end of each symbol would only include the crate name and metadata
from the command line. Metadata from the command line will be passed via a new
command line flag, `-C metadata=foo`, which specifies a string to hash.

## The standard rust distribution

The standard distribution would continue to put hashes in filenames manually
because the libraries are intended to occupy a privileged space on the system.
The build system would manually move a file after it was compiled to the correct
destination filename.

# Drawbacks

* The compiler is able to operate fairly well independently of cargo today, and
  this scheme would hamstring the compiler by limiting the number of "it just
  works" use cases. If cargo is not being used, build systems will likely have
  to start using `--extern` to specify dependencies if name conflicts or version
  conflicts arise between crates.

* This scheme still has redundancy in the list of dependencies with the external
  cargo manifest. The source code would no longer list versions, but the cargo
  manifest will contain the same identifier for each dependency that the source
  code will contain.

# Alternatives

* The compiler could go in the opposite direction of this proposal, enhancing
  `extern crate` instead of simplifying it. The compiler could learn about
  things like version ranges and friends, while still maintaining flags to fine
  tune its behavior. It is unclear whether this increase in complexity will be
  paired with a large enough gain in usability of the compiler independent of
  cargo.

# Unresolved questions

* An implementation for the more advanced features of cargo does not currently
  exist, to it is unknown whether `--extern` will be powerful enough for cargo
  to satisfy all its use cases with.

* Are the string literal parts of `extern crate` justified? Allowing a string
  literal just for the `-` character may be overkill.
