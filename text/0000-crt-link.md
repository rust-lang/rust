- Feature Name: `crt_link`
- Start Date: 2016-08-18
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Enable the compiler to select whether a target dynamically or statically links
to a platform's standard C runtime through the introduction of three orthogonal
and otherwise general purpose features, one of which would likely never become
stable.

# Motivation
[motivation]: #motivation

Today all targets of rustc hard-code how they link to the native C runtime. For
example the `x86_64-unknown-linux-gnu` target links to glibc dynamically,
`x86_64-unknown-linux-musl` links statically to musl, and
`x86_64-pc-windows-msvc` links dynamically to MSVCRT. There are many use cases,
however, where these decisions are not suitable. For example binaries on Alpine
Linux want to link dynamically to musl and redistributable binaries on Windows
are best done by linking statically to MSVCRT.

The actual underlying code essentially never needs to change depending on how
the C runtime is being linked, just the mechanics of how it's actually all
linked together. As a result it's a common desire to take the target libraries
"off the shelf" and change how the C runtime is linked in as well.

The purpose of this RFC is to provide a cross-platform solution spanning both
Cargo and the compiler which allows configuration of how the C runtime is
linked. The idea is that the standard MSVC and musl targets can be used as they
are today with an extra compiler flag to change how the C runtime is linked by
default.

This RFC does *not* propose unifying how the C runtime is linked across
platforms (e.g. always dynamically or always statically) but instead leaves that
decision to each target.

# Detailed design
[design]: #detailed-design

This RFC proposed introducing three separate features to the compiler and Cargo.
When combined together they will enable the compiler to change whether the C
standard library is linked dynamically or statically, but in isolation each
should be useful in its own right.

### A `crt_link` cfg directive

The compiler will first define a new `crt_link` `#[cfg]` directive. This
directive will behave similarly to directives like `target_os` where they're
defined by the compiler for all targets. The compiler will set this value to
either `"dynamic"` or `"static"` depending on how the C runtime is requested to
being linked.

For example, crates can then indicate:

```rust
#[cfg_attr(crt_link = "static", link(name = "c", kind = "static"))]
#[cfg_attr(crt_link = "dynamic", link(name = "c"))]
extern {
    // ...
}
```

This will notably be used in the `libc` crate where the linkage to the C
runtime is defined.

Finally, the compiler will *also* allow defining this attribute from the
command line. For example:

```
rustc --cfg 'crt_link = "static"' foo.rs
```

This will override the compiler's default definition of `crt_link` and use this
one instead. Again, though, the only valid values for this directive are
`"static"` and `"dynamic"`.

In isolation, however, this directive is not too useful, It would still require
rebuilding the `libc` crate (which the standard library links to) if the
linkage to the C runtime needs to change. This is where the two other features
this RFC proposes come into play though!

### Forwarding `#[cfg]` to build scripts

The first feature proposed is enabling Cargo to forward `#[cfg]` directives from
the compiler into build scripts. Currently the compiler supports `--print cfg`
as a flag to print out internal cfg directives, which Cargo currently uses to
implement platform-specific dependencies.

When Cargo runs a build script it already sets a [number of environment
variables][cargo-build-env], and it will now set a family of `CARGO_CFG_*`
environment variables as well. For each key printed out from `rustc --print
cfg`, Cargo will set an environment variable for the build script to learn
about.

[cargo-build-env]: http://doc.crates.io/environment-variables.html#environment-variables-cargo-sets-for-build-scripts

For example, locally `rustc --print cfg` prints:

```
target_os="linux"
target_family="unix"
target_arch="x86_64"
target_endian="little"
target_pointer_width="64"
target_env="gnu"
unix
debug_assertions
```

And with this Cargo would set the following environment variables for build
script invocations for this target.

```
export CARGO_CFG_TARGET_OS=linux
export CARGO_CFG_TARGET_FAMILY=unix
export CARGO_CFG_TARGET_ARCH=x86_64
export CARGO_CFG_TARGET_ENDIAN=little
export CARGO_CFG_TARGET_POINTER_WIDTH=64
export CARGO_CFG_TARGET_ENV=gnu
export CARGO_CFG_UNIX
export CARGO_CFG_DEBUG_ASSERTIONS
```

As mentioned in the previous section, the linkage of the C standard library
will be a `#[cfg]` directive defined by the compiler, and through this method
build scripts will be able to learn how the C standard library is being linked.
This is crucially important for the MSVC target where code needs to be compiled
differently depending on how the C library is linked.

This feature ends up having the added benefit of informing build scripts about
selected CPU features as well. For example once the `target_feature` `#[cfg]`
is stabilized build scripts will know whether SSE/AVX/etc are enabled features
for the C code they might be compiling.

### "Lazy Linking"

The final feature that will be added to the compiler is the ability to "lazily"
link a native library depending on values of `#[cfg]` at compile time of
downstream crates, not of the crate with the `#[link]` directives. This feature
is never intended to be stabilized, and is instead targeted at being an unstable
implementation detail of the `libc` crate.

Specifically, the `#[link]` attribute will be extended with a new directive
that it accepts, `cfg(..)`, such as:

```rust
#[link(name = "foo", cfg(bar))]
```

This `cfg` indicates to the compiler that the `#[link]` annotation only applies
if the `bar` directive is matched. The compiler will then use this knowledge
in two ways:

* When `dllimport` or `dllexport` needs to be applied, it will evaluate the
  current compilation's `#[cfg]` directives and see if upstream `#[link]`
  directives apply or not.

* When deciding what native libraries should be linked, the compiler will
  evaluate whether they should be linked or not depending on the current
  compilation's `#[cfg]` directives nad the upstream `#[link]` directives.

### Customizing linkage to the C runtime

With the above features, the following changes will be made to enable selecting
the linkage of the C runtime at compile time for downstream crates.

First, the `libc` crate will be modified to contain blocks along the lines of:

```rust
cfg_if! {
    if #[cfg(target_env = "musl")] {
        #[link(name = "c", cfg(crt_link = "static"), kind = "static")]
        #[link(name = "c", cfg(crt_link = "dynamic"))]
        extern {}
    } else if #[cfg(target_env = "msvc")] {
        #[link(name = "msvcrt", cfg(crt_link = "dynamic"))]
        #[link(name = "libcmt", cfg(crt_link = "static"))]
        extern {}
    } else {
        // ...
    }
}
```

This informs the compiler that for the musl target if the CRT is statically
linked then the library named `c` is included statically in libc.rlib. If the
CRT is linked dynamically, however, then the library named `c` will be linked
dynamically. Similarly for MSVC, a static CRT implies linking to `libcmt` and a
dynamic CRT implies linking to `msvcrt` (as we do today).

After this change, the gcc-rs crate will be modified to check for the
`CARGO_CFG_CRT_LINK` directive. If it is not present or value is `dynamic`, then
it will compile C code with `/MD`. Otherwise if the value is `static` it will
compile code with `/MT`.

Finally, an example of compiling for MSVC linking statically to the C runtime
would look like:

```
RUSTFLAGS='--cfg crt_link="static"' cargo build --target x86_64-pc-windows-msvc
```

and similarly, compiling for musl but linking dynamically to the C runtime would
look like:

```
RUSTFLAGS='--cfg crt_link="dynamic"' cargo build --target x86_64-unknown-linux-musl
```

### Future work

The features proposed here are intended to be the absolute bare bones of support
needed to configure how the C runtime is linked. A primary drawback, however, is
that it's somewhat cumbersome to select the non-default linkage of the CRT.
Similarly, however, it's cumbersome to select target CPU features which are not
the default, and these two situations are very similar. Eventually it's intended
that there's an ergonomic method for informing the compiler and Cargo of all
"compilation codegen options" over the usage of `RUSTFLAGS` today. It's assume
that configuration of `crt_link` will be included in this ergonomic
configuration as well.

Furthermore, it would have arguably been a "more correct" choice for Rust to by
default statically link to the CRT on MSVC rather than dynamically. While this
would be a breaking change today due to how C components are compiled, if this
RFC is implemented it should not be a breaking change to switch the defaults in
the future.

# Drawbacks
[drawbacks]: #drawbacks

* Working with `RUSTFLAGS` can be cumbersome, but as explained above it's
  planned that eventually there's a much more ergonomic configuration method for
  other codegen options like `target-cpu` which would also encompass the linkage
  of the CRT.

* Adding a feature which is intended to never be stable (`#[link(.., cfg(..))]`)
  is somewhat unfortunate but allows sidestepping some of the more thorny
  questions with how this works. The stable *semantics* will be that for some
  targets the `--cfg crt_link=...` directive affects the linkage of the CRT,
  which seems like a worthy goal regardless.

# Alternatives
[alternatives]: #alternatives

* One alternative is to add entirely new targets, for example
  `x86_64-pc-windows-msvc-static`. Unfortunately though we don't have a great
  naming convention for this, and it also isn't extensible to other codegen
  options like `target-cpu`. Additionally, adding a new target is a pretty
  heavyweight solution as we'd have to start distributing new artifacts and
  such.

* Another possibility would be to start storing metdata in the "target name"
  along the lines of `x86_64-pc-windows-msvc+static`. This is a pretty big
  design space, though, which may not play well with Cargo and build scripts, so
  for now it's preferred to avoid this rabbit hole of design if possible.

* Finally, the compiler could simply have an environment variable which
  indicates the CRT linkage. This would then be read by the compiler and by
  build scripts, and the compiler would have its own back channel for changing
  the linkage of the C library along the lines of `#[link(.., cfg(..))]` above.

# Unresolved questions
[unresolved]: #unresolved-questions

None, yet.
