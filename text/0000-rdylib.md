- Feature Name: N/A
- Start Date: 2016-02-23
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add a new crate type accepted by the compiler, called `rdylib`, which
corresponds to the behavior of `-C prefer-dynamic` plus `--crate-type dylib`.

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
  function sneed to be exported. This can lead to unnecessarily large Rust
  dynamic libraries in terms of object size as well as missed optimization
  opportunities from knowing that a function is otherwise private.
* We can't run LTO for dylibs because those are intended for end products, not
  intermediate ones like (1) is.

The purpose of this RFC is to solve these drawbacks with a new crate-type to
represent the more rarely used form of dynamic library (rdylibs).

# Detailed design
[design]: #detailed-design

A new crate type will be accepted by the compiler, `rdylib`, which can be passed
as either `--crate-type rdylib` on the command line or via `#![crate_type =
"rdylib"]` in crate attributes. This crate type will conceptually correspond to
the rdylib use case described above, and today's `dylib` crate-type will
correspond to the cdylib use case above. Note that the literal output artifacts
of these two crate types (files, file names, etc) will be the same.

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

As is evidenced from many of these changes, however, the reinterpretation of the
`dylib` output format from what it is today is a breaking change. For example
metadata will not be present and symbols will be hidden. As a result, this RFC
has a...

### Transition Plan

This RFC is technically a breaking change, but it is expected to not actually
break many work flows in practice because there is only one known user of
rdylibs, the compiler itself. This notably means that plugins will also need to
be compiled differently, but because they are nightly-only we've got some more
leeway around them.

All other known users of the `dylib` output crate type fall into the cdylib use
case. The "breakage" here would mean:

* The metadata section no longer exists. In almost all cases this just means
  that the output artifacts will get smaller if it isn't present, it's expected
  that no one other than the compiler itself is actually consuming this
  information.
* Rust symbols will be hidden by default. The symbols, however, have
  unpredictable hashes so there's not really any way they can be meaningfully
  leveraged today.

Given that background, it's expected that if there's a smooth migration path for
plugins and the compiler then the "breakage" here won't actually appear in
practice. The proposed implementation strategy and migration path is:

1. Implement the `rdylib` output type as proposed in this RFC.
2. Change Cargo to use `--crate-type rdylib` when compiling plugins instead of
   `--crate-type dylib` + `-C prefer-dynamic`.
3. Implement the changes to the `dylib` output format as proposed in this RFC.

So long as the steps are spaced apart by a few days it should be the case that
no nightly builds break if they're always using an up-to-date nightly compiler.

# Drawbacks
[drawbacks]: #drawbacks

Rust's ephemeral and ill-defined "linkage model" is... well... ill defined and
ephemeral. This RFC is an extension of this model, but it's difficult to reason
about extending that which is not well defined. As a result there could be
unforseen interactions between this output format and where it's used.

As usual, of course, proposing a breaking change is indeed a drawback. It is
expected that RFC doesn't break anything in practice, but that'd be difficult to
gauge until it's implemented.

# Alternatives
[alternatives]: #alternatives

* Instead of reinterpreting the `dylib` output format as a cdylib, we could
  continue interpreting it as an rdylib and add a new dedicated `cdylib` output
  format. This would not be a breaking change, but it doesn't come without its
  drawbacks. As the most common output type, many projects would have to switch
  to `cdylib` from `dylib`, meaning that they no longer support older Rust
  compilers. This may also take time to propagate throughout the community. It's
  also arguably a "better name", so this RFC proposes an
  in-practice-not-a-breaking-change by adding a worse name of `rdylib` for the
  less used output format.

* The compiler could have a longer transition period where `-C prefer-dynamic`
  plus `--crate-type dylib` is interpreted as an rdylib. Either that or the
  implementation strategy here could be extended by a release or two to let
  changes time to propagate throughout the ecosystem.

# Unresolved questions
[unresolved]: #unresolved-questions

* This RFC is currently founded upon the assumption that rdylibs are very rarely
  used in the ecosystem. An audit has not been performed to determine whether
  this is true or not, but is this actually the case?

* Should the new `rdylib` format be considered unstable? (should it require a
  nightly compiler?). The use case for a Rust dynamic library is so limited, and
  so volatile, we may want to just gate access to it by default.
