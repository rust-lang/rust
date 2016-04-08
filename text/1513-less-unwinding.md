- Feature Name: `panic_runtime`
- Start Date: 2016-02-25
- RFC PR: https://github.com/rust-lang/rfcs/pull/1513
- Rust Issue: https://github.com/rust-lang/rust/issues/32837

# Summary
[summary]: #summary

Stabilize implementing panics as aborts.

* Stabilize the `-Z no-landing-pads` flag under the name `-C panic=strategy`
* Implement a number of unstable features akin to custom allocators to swap out
  implementations of panic just before a final product is generated.
* Add a `[profile.dev]` option to Cargo to configure how panics are implemented.

# Motivation
[motivation]: #motivation

Panics in Rust have long since been implemented with the intention of being
caught at particular boundaries (for example the thread boundary). This is quite
useful for isolating failures in Rust code, for example:

* Servers can avoid taking down the entire process but can instead just take
  down one request.
* Embedded Rust libraries can avoid taking down the entire process and can
  instead gracefully inform the caller that an internal logic error occurred.
* Rust applications can isolate failure from various components. The classical
  example of this is Servo can display a "red X" for an image which fails to
  decode instead of aborting the entire browser or killing an entire page.

While these are examples where a recoverable panic is useful, there are many
applications where recovering panics is undesirable or doesn't lead to anything
productive:

* Rust applications which use `Result` for error handling typically use `panic!`
  to indicate a fatal error, in which case the process *should* be taken down.
* Many applications simply can't recover from an internal assertion failure, so
  there's no need trying to recover it.
* To implement a recoverable panic, the compiler and standard library use a
  method called stack unwinding. The compiler must generate code to support this
  unwinding, however, and this takes time in codegen and optimizers.
* Low-level applications typically don't use unwinding at all as there's no
  stack unwinder (e.g. kernels).

> **Note**: as an idea of the compile-time and object-size savings from
> disabling the extra codegen, compiling Cargo as a library is 11% faster (16s
> from 18s) and 13% smaller (15MB to 13MB). Sizable gains!

Overall, the ability to recover panics is something that needs to be decided at
the application level rather than at the language level. Currently the compiler
does not support the ability to translate panics to process aborts in a stable
fashion, and the purpose of this RFC is to add such a venue.

With such an important codegen option, however, as whether or not exceptions can
be caught, it's easy to get into a situation where libraries of mixed
compilation modes are linked together, causing odd or unknown errors. This RFC
proposes a situation similar to the design of custom allocators to alleviate
this situation.

# Detailed design
[design]: #detailed-design

The major goal of this RFC is to develop a work flow around managing crates
which wish to disable unwinding. This intends to set forth a complete vision for
how these crates interact with the ecosystem at large. Much of this design will
be similar to the [custom allocator RFC][custom-allocators].

[custom-allocators]: https://github.com/rust-lang/rfcs/blob/master/text/1183-swap-out-jemalloc.md

### High level design

This section serves as a high-level tour through the design proposed in this
RFC. The linked sections provide more complete explanation as to what each step
entails.

* The compiler will have a [new stable flag](#new-compiler-flags), `-C panic`
  which will configure how unwinding-related code is generated.
* [Two new unstable attributes](#panic-attributes) will be added to the
  compiler, `#![needs_panic_runtime]` and `#![panic_runtime]`. The standard
  library will need a runtime and will be lazily linked to a crate which has
  `#![panic_runtime]`.
* [Two unstable crates](#panic-crates) tagged with `#![panic_runtime]` will be
  distributed as the runtime implementation of panicking, `panic_abort` and
  `panic_unwind` crates.  The former will translate all panics to process
  aborts, whereas the latter will be implemented as unwinding is today, via the
  system stack unwinder.
* [Cargo will gain](#cargo-changes) a new `panic` option in the `[profile.foo]`
  sections to indicate how that profile should compile panic support.

### New Compiler Flags

The first component to this design is to have a **stable** flag to the compiler
which configures how panic-related code is generated.  This will be
stabilized in the form:

```
$ rustc -C help

Available codegen options:

    ...
    -C             panic=val -- strategy to compile in for panic related code
    ...
```

There will currently be two supported strategies:

* `unwind` - this is what the compiler implements by default today via the
  `invoke` LLVM instruction.
* `abort` - this will implement that `-Z no-landing-pads` does today, which is
  to disable the `invoke` instruction and use `call` instead everywhere.

This codegen option will default to `unwind` if not specified (what happens
today), and the value will be encoded into the crate metadata. This option is
planned with extensibility in mind to future panic strategies if we ever
implement some (return-based unwinding is at least one other possible option).

### Panic Attributes

Very similarly to [custom allocators][allocator-attributes], two new
**unstable** crate attributes will be added to the compiler:

[allocator-attributes]: https://github.com/rust-lang/rfcs/blob/master/text/1183-swap-out-jemalloc.md#new-attributes

* `#![needs_panic_runtime]` - indicates that this crate requires a "panic
  runtime" to link correctly. This will be attached to the standard library and
  is not intended to be attached to any other crate.
* `#![panic_runtime]` - indicates that this crate is a runtime implementation of
  panics.

As with allocators, there are a number of limitations imposed by these
attributes by the compiler:

* Any crate DAG can only contain at most one instance of `#![panic_runtime]`.
* Implicit dependency edges are drawn from crates tagged with
  `#![needs_panic_runtime]` to those tagged with `#![panic_runtime]`. Loops as
  usual are forbidden (e.g. a panic runtime can't depend on libstd).
* Complete artifacts which include a crate tagged with `#![needs_panic_runtime]`
  must include a panic runtime. This includes executables, dylibs, and
  staticlibs. If no panic runtime is explicitly linked, then the compiler will
  select an appropriate runtime to inject.
* Finally, the compiler will ensure that panic runtimes and compilation modes
  are not mismatched. For a final product (outputs that aren't rlibs) the
  `-C panic` mode of the panic runtime must match the final product itself. If
  the panic mode is `abort`, then no other validation is performed, but
  otherwise all crates in the DAG must have the same value of `-C panic`.

The purpose of these limitations is to solve a number of problems that arise
when switching panic strategies. For example with aborting panic crates won't
have to link to runtime support of unwinding, or rustc will disallow mixing
panic strategies by accident.

The actual API of panic runtimes will not be detailed in this RFC. These new
attributes will be unstable, and consequently the API itself will also be
unstable. It suffices to say, however, that like custom allocators a panic
runtime will implement some public `extern` symbols known to the crates that
need a panic runtime, and that's how they'll communicate/link up.

### Panic Crates

Two new **unstable** crates will be added to the distribution for each target:

* `panic_unwind` - this is an extraction of the current implementation of
  panicking from the standard library. It will use the same mechanism of stack
  unwinding as is implemented on all current platforms.
* `panic_abort` - this is a new implementation of panicking which will simply
  translate unwinding to process aborts. There will be no runtime support
  required by this crate.

The compiler will assume that these crates are distributed for each platform
where the standard library is also distributed (e.g. a crate that has
`#![needs_panic_runtime]`).

### Compiler defaults

The compiler will ship with a few defaults which affect how panic runtimes are
selected in Rust programs. Specifically:

* The `-C panic` option will default to **unwind** as it does today.
* The libtest crate will explicitly link to `panic_unwind`. The test runner that
  libtest implements relies on equating panics with failure and cannot work if
  panics are translated to aborts.
* If no panic runtime is explicitly selected, the compiler will employ the
  following logic to decide what panic runtime to inject:

  1. If any crate in the DAG is compiled with `-C panic=abort`, then `panic_abort`
     will be injected.
  2. If all crates in the DAG are compiled with `-C panic=unwind`, then
     `panic_unwind` is injected.

### Cargo changes

In order to export this new feature to Cargo projects, a new option will be
added to the `[profile]` section of manifests:

```toml
[profile.dev]
panic = 'unwind'
```

This will cause Cargo to pass `-C panic=unwind` to all `rustc` invocations for
a crate graph. Cargo will have special knowledge, however, that for `cargo
test` it cannot pass `-C panic=abort`.

# Drawbacks
[drawbacks]: #drawbacks

* The implementation of custom allocators was no small feat in the compiler, and
  much of this RFC is essentially the same thing. Similar infrastructure can
  likely be leveraged to alleviate the implementation complexity, but this is
  undeniably a large change to the compiler for albeit a relatively minor
  option. The counter point to this, however, is that disabling unwinding in a
  principled fashion provides far higher quality error messages, prevents
  erroneous situations, and provides an immediate benefit for many Rust users
  today.

* The binary distribution of the standard library will not change from what it
  is today. In other words, the standard library (and dependency crates like
  libcore) will be compiled with `-C panic=unwind`. This introduces the
  opportunity for extra code bloat or missed optimizations in applications that
  end up disabling unwinding in the long run. Distribution, however, is *far*
  easier because there's only one copy of the standard library and we don't have
  to rely on any other form of infrastructure.

* This represents a proliferation of the `#![needs_foo]` and `#![foo]` style
  system that allocators have begun. This may be indicative of a deeper
  underlying requirement here of the standard library or perhaps showing how the
  strategy in the standard library needs to change. If the standard library were
  a crates.io crate it would arguably support these options via Cargo features,
  but without that option is this the best way to be implementing these switches
  for the standard library?

# Alternatives
[alternatives]: #alternatives

* Currently this RFC allows mixing multiple panic runtimes in a crate graph so
  long as the actual runtime is compiled with `-C panic=abort`. This is
  primarily done to immediately reap benefit from `-C panic=abort` even though
  the standard library we distribute will still have unwinding support compiled
  in (compiled with `-C panic=unwind`). In the not-too-distant future however,
  we will likely be poised to distribute multiple binary copies of the standard
  library compiled with different profiles. We may be able to tighten this
  restriction on behalf of the compiler, requiring that all crates in a DAG have
  the same `-C panic` compilation mode, but there would unfortunately be no
  immediate benefit to implementing the RFC from users of our precompiled
  nightlies.

  This alternative, additionally, can also be viewed as a drawback. It's unclear
  what a future libstd distribution mechanism would look like and how this RFC
  might interact with it. Stabilizing disabling unwinding via a compiler switch
  or a Cargo profile option may not end up meshing well with the strategy we
  pursue with shipping multiple standard libraries.

* Instead of the panic runtime support in this RFC, we could instead just ship
  two different copies of the standard library where one simply translates
  panics to abort instead of unwinding. This is unfortunately very difficult
  for Cargo or the compiler to track, however, to ensure that the codegen
  option of how panics are translated is propagated throughout the rest of
  the crate graph. Additionally it may be easy to mix up crates of different
  panic strategies.

# Unresolved questions
[unresolved]: #unresolved-questions

* One possible implementation of unwinding is via return-based flags. Much of
  this RFC is designed with the intention of supporting arbitrary unwinding
  implementations, but it's unclear whether it's too heavily biased towards
  panic is either unwinding or aborting.

* The current implementation of Cargo would mean that a naive implementation of
  the profile option would cause recompiles between `cargo build` and `cargo
  test` for projects that specify `panic = 'abort'`. Is this acceptable? Should
  Cargo cache both copies of the crate?
