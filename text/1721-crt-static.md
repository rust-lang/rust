- Feature Name: `crt_link`
- Start Date: 2016-08-18
- RFC PR: [rust-lang/rfcs#1721](https://github.com/rust-lang/rfcs/pull/1721)
- Rust Issue: [rust-lang/rust#37406](https://github.com/rust-lang/rust/issues/37406)

# Summary
[summary]: #summary

Enable the compiler to select whether a target dynamically or statically links
to a platform's standard C runtime through the introduction of three orthogonal
and otherwise general purpose features, one of which will likely never become
stable and can be considered an implementation detail of std. These features do
not require the compiler or language to have intrinsic knowledge of the
existence of C runtimes.

The end result is that rustc will be able to reuse its existing standard library
binaries for the MSVC and musl targets to build code that links either
statically or dynamically to libc.

The design herein additionally paves the way for improved support for
dllimport/dllexport, and cpu-specific features, particularly when
combined with a [std-aware cargo].

[std-aware cargo]: https://github.com/rust-lang/rfcs/pull/1133

# Motivation
[motivation]: #motivation

Today all targets of rustc hard-code how they link to the native C runtime. For
example the `x86_64-unknown-linux-gnu` target links to glibc dynamically,
`x86_64-unknown-linux-musl` links statically to musl, and
`x86_64-pc-windows-msvc` links dynamically to MSVCRT. There are many use cases,
however, where these decisions are not suitable. For example binaries on Alpine
Linux want to link dynamically to musl and creating portable binaries on Windows
is most easily done by linking statically to MSVCRT.

Today rustc has no mechanism for accomplishing this besides defining an entirely
new target specification and distributing a build of the standard library for
it. Because target specifications must be described by a target triple, and
target triples have preexisting conventions into which such a scheme does not
fit, we have resisted doing so.

# Detailed design
[design]: #detailed-design

This RFC introduces three separate features to the compiler and Cargo. When
combined they will enable the compiler to change whether the C standard library
is linked dynamically or statically. In isolation each feature is a natural
extension of existing features, and each should be useful on its own.

A key insight is that, for practical purposes, the object code _for the standard
library_ does not need to change based on how the C runtime is being linked;
though it is true that on Windows, it is _generally_ important to properly
manage the use of dllimport/dllexport attributes based on the linkage type, and
C code does need to be compiled with specific options based on the linkage type.
So it is technically possible to produce Rust executables and dynamic libraries
that either link to libc statically or dynamically from a single std binary by
correctly manipulating the arguments to the linker.

A second insight is that there are multiple existing, unserved use cases for
configuring features of the hardware architecture, underlying platform, or
runtime [1], which require the entire 'world', possibly including std, to be
compiled a certain way. C runtime linkage is another example of this
requirement.

[1]: https://internals.rust-lang.org/t/pre-rfc-a-vision-for-platform-architecture-configuration-specific-apis/3502

From these observations we can design a cross-platform solution spanning both
Cargo and the compiler by which Rust programs may link to either a dynamic or
static C library, using only a single std binary. As future work this RFC
discusses how the proposed scheme scheme can be extended to rebuild std
specifically for a particular C-linkage scenario, which may have minor
advantages on Windows due to issues around dllimport and dllexport; and how this
scheme naturally extends to recompiling std in the presence of modified CPU
features.

This RFC does *not* propose unifying how the C runtime is linked across
platforms (e.g. always dynamically or always statically) but instead leaves that
decision to each target, and to future work.

In summary the new mechanics are:

- Specifying C runtime linkage via `-C target-feature=+crt-static` or `-C
  target-feature=-crt-static`. This extends `-C target-feature` to mean not just
  "CPU feature" ala LLVM, but "feature of the Rust target". Several existing
  properties of this flag, the ability to add, with `+`, _or remove_, with `-`,
  the feature, as well as the automatic lowering to `cfg` values, are crucial to
  later aspects of the design. This target feature will be added to targets via
  a small extension to the compiler's target specification.
- Lowering `cfg` values to Cargo build script environment variables. This will
  enable build scripts to understand all enabled features of a target (like
  `crt-static` above) to, for example, compile C code correctly on MSVC.
- Lazy link attributes. This feature is only required by std's own copy of the
  libc crate, and only because std is distributed in binary form and it may yet
  be a long time before Cargo itself can rebuild std.

### Specifying dynamic/static C runtime linkage

A new `target-feature` flag will now be supported by the compiler for relevant
targets: `crt-static`. This can be enabled and disabled in the compiler via:

```
rustc -C target-feature=+crt-static ...
rustc -C target-feature=-crt-static ...
```

Currently all `target-feature` flags are passed through straight to LLVM, but
this proposes extending the meaning of `target-feature` to Rust-target-specific
features as well. Target specifications will be able to indicate what custom
target-features can be defined, and most existing targets will define a new
`crt-static` feature which is turned off by default (except for musl).

The default of `crt-static` will be different depending on the target. For
example `x86_64-unknown-linux-musl` will have it on by default, whereas
`arm-unknown-linux-musleabi` will have it turned off by default.

### Lowering `cfg` values to Cargo build script environment variables

Cargo will begin to forward `cfg` values from the compiler into build
scripts. Currently the compiler supports `--print cfg` as a flag to print out
internal cfg directives, which Cargo uses to implement platform-specific
dependencies.

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

As mentioned in the previous section, the linkage of the C standard library will
be specified as a target feature, which is lowered to a `cfg` value, thus giving
build scripts the ability to modify compilation options based on C standard
library linkage. One important complication here is that `cfg` values in Rust
may be defined multiple times, and this is the case with target features. When a
`cfg` value is defined multiple times, Cargo will create a single environment
variable with a comma-separated list of values.

So for a target with the following features enabled

```
target_feature="sse"
target_feature="crt-static"
```

Cargo would convert it to the following environment variable:

```
export CARGO_CFG_TARGET_FEATURE=sse,crt-static
```

Through this method build scripts will be able to learn how the C standard
library is being linked.  This is crucially important for the MSVC target where
code needs to be compiled differently depending on how the C library is linked.

This feature ends up having the added benefit of informing build scripts about
selected CPU features as well. For example once the `target_feature` `#[cfg]`
is stabilized build scripts will know whether SSE/AVX/etc are enabled features
for the C code they might be compiling.

After this change, the gcc-rs crate will be modified to check for the
`CARGO_CFG_TARGET_FEATURE` directive, and parse it into a list of enabled
features. If the `crt-static` feature is not enabled it will compile C code on
the MSVC target with `/MD`, indicating dynamic linkage. Otherwise if the value
is `static` it will compile code with `/MT`, indicating static linkage. Because
today the MSVC targets use dynamic linkage and gcc-rs compiles C code with `/MD`,
gcc-rs will remain forward and backwards compatible with existing and future
Rust MSVC toolchains until such time as the the decision is made to change the
MSVC toolchain to `+crt-static` by default.

### Lazy link attributes

The final feature that will be added to the compiler is the ability to "lazily"
interpret the linkage requirements of a native library depending on values of
`cfg` at compile time of downstream crates, not of the crate with the `#[link]`
directives. This feature is never intended to be stabilized, and is instead
targeted at being an unstable implementation detail of the `libc` crate linked
to `std` (but _not_ the stable `libc` crate deployed to crates.io).

Specifically, the `#[link]` attribute will be extended with a new argument
that it accepts, `cfg(..)`, such as:

```rust
#[link(name = "foo", cfg(bar))]
```

This `cfg` indicates to the compiler that the `#[link]` annotation only applies
if the `bar` directive is matched. This interpretation is done not during
compilation of the crate in which the `#[link]` directive appears, but during
compilation of the crate in which linking is finally performed. The compiler
will then use this knowledge in two ways:

* When `dllimport` or `dllexport` needs to be applied, it will evaluate the
  final compilation unit's `#[cfg]` directives and see if upstream `#[link]`
  directives apply or not.

* When deciding what native libraries should be linked, the compiler will
  evaluate whether they should be linked or not depending on the final
  compilation's `#[cfg]` directives and the upstream `#[link]` directives.

### Customizing linkage to the C runtime

With the above features, the following changes will be made to select the
linkage of the C runtime at compile time for downstream crates.

First, the `libc` crate will be modified to contain blocks along the lines of:

```rust
cfg_if! {
    if #[cfg(target_env = "musl")] {
        #[link(name = "c", cfg(target_feature = "crt-static"), kind = "static")]
        #[link(name = "c", cfg(not(target_feature = "crt-static")))]
        extern {}
    } else if #[cfg(target_env = "msvc")] {
        #[link(name = "msvcrt", cfg(not(target_feature = "crt-static")))]
        #[link(name = "libcmt", cfg(target_feature = "crt-static"))]
        extern {}
    } else {
        // ...
    }
}
```

This informs the compiler that, for the musl target, if the CRT is statically
linked then the library named `c` is included statically in libc.rlib. If the
CRT is linked dynamically, however, then the library named `c` will be linked
dynamically. Similarly for MSVC, a static CRT implies linking to `libcmt` and a
dynamic CRT implies linking to `msvcrt` (as we do today).

Finally, an example of compiling for MSVC and linking statically to the C
runtime would look like:

```
RUSTFLAGS='-C target-feature=+crt-static' cargo build --target x86_64-pc-windows-msvc
```

and similarly, compiling for musl but linking dynamically to the C runtime would
look like:

```
RUSTFLAGS='-C target-feature=-crt-static' cargo build --target x86_64-unknown-linux-musl
```

### Future work

The features proposed here are intended to be the absolute bare bones of support
needed to configure how the C runtime is linked. A primary drawback, however, is
that it's somewhat cumbersome to select the non-default linkage of the CRT.
Similarly, however, it's cumbersome to select target CPU features which are not
the default, and these two situations are very similar. Eventually it's intended
that there's an ergonomic method for informing the compiler and Cargo of all
"compilation codegen options" over the usage of `RUSTFLAGS` today.

Furthermore, it would have arguably been a "more correct" choice for Rust to by
default statically link to the CRT on MSVC rather than dynamically. While this
would be a breaking change today due to how C components are compiled, if this
RFC is implemented it should not be a breaking change to switch the defaults in
the future, after a reasonable transition period.

The support in this RFC implies that the exact artifacts that we're shipping
will be usable for both dynamically and statically linking the CRT.
Unfortunately, however, on MSVC code is compiled differently if it's linking to
a dynamic library or not. The standard library uses very little of the MSVCRT,
so this won't be a problem in practice for now, but runs the risk of binding our
hands in the future. It's intended, though, that Cargo [will eventually support
custom-compiling the standard library][std-aware cargo]. The `crt-static`
feature would simply be another input to this logic, so Cargo would
custom-compile the standard library if it differed from the upstream artifacts,
solving this problem.

### References

- [Issue about MSVCRT static linking]
  (https://github.com/rust-lang/libc/issues/290)
- [Issue about musl dynamic linking]
  (https://github.com/rust-lang/rust/issues/34987)
- [Discussion on issues around glgobal codegen configuration]
  (https://internals.rust-lang.org/t/pre-rfc-a-vision-for-platform-architecture-configuration-specific-apis/3502)
- [std-aware Cargo RFC]
  (https://github.com/rust-lang/libc/issues/290).
  A proposal to teach Cargo to build the standard library. Rebuilding of std will
  likely in the future be influenced by `-C target-feature`.
- [Cargo's documentation on build-script environment variables]
  (https://github.com/rust-lang/libc/issues/290)

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

* The lazy semantics of `#[link(cfg(..))]` are not so obvious from the name (no
  other `cfg` attribute is treated this way). But this seems a minor issue since
  the feature serves one implementation-specif purpose and isn't intended for
  stabilization.

# Alternatives
[alternatives]: #alternatives

* One alternative is to add entirely new targets, for example
  `x86_64-pc-windows-msvc-static`. Unfortunately though we don't have a great
  naming convention for this, and it also isn't extensible to other codegen
  options like `target-cpu`. Additionally, adding a new target is a pretty
  heavyweight solution as we'd have to start distributing new artifacts and
  such.

* Another possibility would be to start storing metadata in the "target name"
  along the lines of `x86_64-pc-windows-msvc+static`. This is a pretty big
  design space, though, which may not play well with Cargo and build scripts, so
  for now it's preferred to avoid this rabbit hole of design if possible.

* Finally, the compiler could simply have an environment variable which
  indicates the CRT linkage. This would then be read by the compiler and by
  build scripts, and the compiler would have its own back channel for changing
  the linkage of the C library along the lines of `#[link(.., cfg(..))]` above.

* Another approach has [been proposed recently][rfc-1684] that has
  rustc define an environment variable to specify the C runtime kind.

[rfc-1684]: https://github.com/rust-lang/rfcs/pull/1684

* Instead of extending the semantics of `-C target-feature` beyond "CPU
  features", we could instead add a new flag for the purpose, e.g. `-C
  custom-feature`.

# Unresolved questions
[unresolved]: #unresolved-questions

* What happens during the `cfg` to environment variable conversion for values
  that contain commas? It's an unusual corner case, and build scripts should not
  depend on such values, but it needs to be handled sanely.

* Is it really true that lazy linking is only needed by std's libc? What about
  in a world where we distribute more precompiled binaries than just std?

