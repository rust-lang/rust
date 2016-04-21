- Feature Name: N/A
- Start Date: 2016-02-23
- RFC PR: [rust-lang/rfcs#1510](https://github.com/rust-lang/rfcs/pull/1510)
- Rust Issue: [rust-lang/rust#33132](https://github.com/rust-lang/rust/issues/33132)

# Summary
[summary]: #summary

Add a new crate type accepted by the compiler, called `cdylib`, which
corresponds to exporting a C interface from a Rust dynamic library.

# Motivation
[motivation]: #motivation

Currently the compiler supports two modes of generating dynamic libraries:

1. One form of dynamic library is intended for reuse with further compilations.
   This kind of library exposes all Rust symbols, links to the standard library
   dynamically, etc. I'll refer to this mode as **rdylib** as it's a Rust
   dynamic library talking to Rust.
2. Another form of dynamic library is intended for embedding a Rust application
   into another. Currently the only difference from the previous kind of dynamic
   library is that it favors linking statically to other Rust libraries
   (bundling them inside). I'll refer to this as a **cdylib** as it's a Rust
   dynamic library exporting a C API.

Each of these flavors of dynamic libraries has a distinct use case. For examples
rdylibs are used by the compiler itself to implement plugins, and cdylibs are
used whenever Rust needs to be dynamically loaded from another language or
application.

Unfortunately the balance of features is tilted a little bit too much towards
the smallest use case, rdylibs. In practice because Rust is statically linked by
default and has an unstable ABI, rdylibs are used quite rarely. There are a
number of requirements they impose, however, which aren't necessary for
cdylibs:

* Metadata is included in all dynamic libraries. If you're just loading Rust
  into somewhere else, however, you have no need for the metadata!
* *Reachable* symbols are exposed from dynamic libraries, but if you're loading
  Rust into somewhere else then, like executables, only *public* non-Rust-ABI
  functions need to be exported. This can lead to unnecessarily large Rust
  dynamic libraries in terms of object size as well as missed optimization
  opportunities from knowing that a function is otherwise private.
* We can't run LTO for dylibs because those are intended for end products, not
  intermediate ones like (1) is.

The purpose of this RFC is to solve these drawbacks with a new crate-type to
represent the more rarely used form of dynamic library (rdylibs).

# Detailed design
[design]: #detailed-design

A new crate type will be accepted by the compiler, `cdylib`, which can be passed
as either `--crate-type cdylib` on the command line or via `#![crate_type =
"cdylib"]` in crate attributes. This crate type will conceptually correspond to
the cdylib use case described above, and today's `dylib` crate-type will
continue to correspond to the rdylib use case above. Note that the literal
output artifacts of these two crate types (files, file names, etc) will be the
same.

The two formats will differ in the parts listed in the motivation above,
specifically:

* **Metadata** - rdylibs will have a section of the library with metadata,
  whereas cdylibs will not.
* **Symbol visibility** - rdylibs will expose all symbols as rlibs do, cdylibs
  will expose symbols as executables do. This means that `pub fn foo() {}` will
  not be an exported symbol, but `#[no_mangle] pub extern fn foo() {}` will be
  an exported symbol. Note that the compiler will also be at liberty to pass
  extra flags to the linker to actively hide exported Rust symbols from linked
  libraries.
* **LTO** - this will disallowed for rdylibs, but enabled for cdylibs.
* **Linkage** - rdylibs will link dynamically to one another by default, for
  example the standard library will be linked dynamically by default. On the
  other hand, cdylibs will link all Rust dependencies statically by default.

# Drawbacks
[drawbacks]: #drawbacks

Rust's ephemeral and ill-defined "linkage model" is... well... ill defined and
ephemeral. This RFC is an extension of this model, but it's difficult to reason
about extending that which is not well defined. As a result there could be
unforseen interactions between this output format and where it's used.

# Alternatives
[alternatives]: #alternatives

* Originally this RFC proposed adding a new crate type, `rdylib`, instead of
  adding a new crate type, `cdylib`. The existing `dylib` output type would be
  reinterpreted as a cdylib use-case. This is unfortunately, however, a breaking
  change and requires a somewhat complicated transition plan in Cargo for
  plugins. In the end it didn't seem worth it for the benefit of "cdylib is
  probably what you want".

# Unresolved questions
[unresolved]: #unresolved-questions

* Should the existing `dylib` format be considered unstable? (should it require
  a nightly compiler?). The use case for a Rust dynamic library is so limited,
  and so volatile, we may want to just gate access to it by default.
