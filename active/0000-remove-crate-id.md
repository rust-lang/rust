- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Remove the `crate_id` attribute, knowledge of versions from rustc, and the
`crate_type` attribute. Allow keywords in attributes, and add a `#[crate]`
attribute similar to the old `#[link]` attribute. Filenames will no longer have
versions, nor will symbols, and the compiler will grow a new flag, `--extern`,
to override searching for external crates.

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

A new `crate` attribute will be accepted by the compiler. For example:

```rust
#![crate(name = "json", type = "dylib", type = "rlib", version = "1.0.2-pre")]
```

Breaking down this attribute:

* `crate` - This is a new top-level attribute which has an inner list of meta
            items which are intended to describe the current crate.
* `name` - This meta item is recognized by the compiler to provide a name for
           the crate being compiled. This will override the compiler's inference
           of the crate name based on the file name being compiled. This is also
           later used to match with when upstream crates link to this one.
* `type` - This will supersede the `#[crate_type]` attribute. The `crate`
           attribute can contain multiple `type` meta items, describing the type
           of output of this crate. This is used to describe whether the output
           is an rlib, dylib, executable, or staticlib.
* `version` - This, and all other members of the `crate` attribute, are not
              specially recognized by the compiler. These attributes are used as
              controllers for the hashes used in the rest of the compiler.

In addition to allowing data to hash being specified through the `crate`
attribute, the compiler will also grow a new flag, `-C metadata=foo` to be able
to specify arbitrary strings to hash via the command line. The usage of this
hashed data is described below.

## Keywords in attributes

The compiler currently disallows keywords in attributes. This rule would be
amended to allow any identifier an attribute names and attribute keys. This is
primarily done to allow the `crate` and `type` attributes to exist.

## Naming library filenames

Currently, rustc crates filenames for library following this pattern:

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

Under this new scheme, the new filenames would be:

```
lib<name>-<hash>.rlib
```

Where the new hash is defined to be the hash of the contents of the metadata.
The metadata is a merging of the command line metadata as well as the contents
of the `crate` attribute. All three of the original goals are still satisfied by
this new naming scheme. The version would no longer be explicitly visible in the
filename, but the hash would include the version and be distinct across two
versions of the same library.

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
optional (defaulting to the identifier), and the string can only contain
alphanumeric characters, `-`, and `_`. Note that the string is not a valid rust
identifier due to its usage of the `-` character.

The compiler's searching and file matching logic would be altered to only match
crates based on name. If two versions of a crate are found, the compiler will
unconditionally emit an error. It will be up to the user to move the two
libraries on the filesystem and control the `-L` flags to the compiler to enable
disambiguation.

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
hash at the end of each symbol would still include the symbol via the metadata
in the `crate` attribute or from the command line, however.

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

* Does allowing keywords in attributes set an unusual precedent for other
  portions of the language?
