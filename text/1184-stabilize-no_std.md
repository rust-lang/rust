- Feature Name: N/A
- Start Date: 2015-06-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/1184
- Rust Issue: https://github.com/rust-lang/rust/issues/27394

# Summary

Tweak the `#![no_std]` attribute, add a new `#![no_core]` attribute, and
pave the way for stabilizing the libcore library.

# Motivation

Currently all stable Rust programs must link to the standard library (libstd),
and it is impossible to opt out of this. The standard library is not appropriate
for use cases such as kernels, embedded development, or some various niche cases
in userspace. For these applications Rust itself is appropriate, but the
compiler does not provide a stable interface compiling in this mode.

The standard distribution provides a library, libcore, which is "the essence of
Rust" as it provides many language features such as iterators, slice methods,
string methods, etc. The defining feature of libcore is that it has 0
dependencies, unlike the standard library which depends on many I/O APIs, for
example. The purpose of this RFC is to provide a stable method to access
libcore.

Applications which do not want to use libstd still want to use libcore 99% of
the time, but unfortunately the current `#![no_std]` attribute does not do a
great job in facilitating this. When moving into the realm of not using the
standard library, the compiler should make the use case as ergonomic as
possible, so this RFC proposes different behavior than today's `#![no_std]`.

Finally, the standard library defines a number of language items which must be
defined when libstd is not used. These language items are:

* `panic_fmt`
* `eh_personality`
* `stack_exhausted`

To be able to usefully leverage `#![no_std]` in stable Rust these lang items
must be available in a stable fashion.

# Detailed Design

This RFC proposes a nuber of changes:

* Tweak the `#![no_std]` attribute slightly.
* Introduce a `#![no_core]` attribute.
* Pave the way to stabilize the `core` module.

## `no_std`

The `#![no_std]` attribute currently provides two pieces of functionality:

* The compiler no longer injects `extern crate std` at the top of a crate.
* The prelude (`use std::prelude::v1::*`) is no longer injected at the top of
  every module.

This RFC proposes adding the following behavior to the `#![no_std]` attribute:

* The compiler will inject `extern crate core` at the top of a crate.
* The libcore prelude will be injected at the top of every module.

Most uses of `#![no_std]` already want behavior along these lines as they want
to use libcore, just not the standard library.

## `no_core`

A new attribute will be added to the compiler, `#![no_core]`, which serves two
purposes:

* This attribute implies the `#![no_std]` attribute (no std prelude/crate
  injection).
* This attribute will prevent core prelude/crate injection.

Users of `#![no_std]` today who do *not* use libcore would migrate to moving
this attribute instead of `#![no_std]`.

## Stabilization of libcore

This RFC does not yet propose a stabilization path for the contents of libcore,
but it proposes readying to stabilize the name `core` for libcore, paving the
way for the rest of the library to be stabilized. The exact method of
stabilizing its contents will be determined with a future RFC or pull requests.

## Stabilizing lang items

As mentioned above, there are three separate lang items which are required by
the libcore library to link correctly. These items are:

* `panic_fmt`
* `stack_exhausted`
* `eh_personality`

This RFC does **not** attempt to stabilize these lang items for a number of
reasons:

* The exact set of these lang items is somewhat nebulous and may change over
  time.
* The signatures of each of these lang items can either be platform-specific or
  it's just "too weird" to stabilize.
* These items are pretty obscure and it's not very widely known what they do or
  how they should be implemented.

Stabilization of these lang items (in any form) will be considered in a future
RFC.

# Drawbacks

The current distribution provides precisely one library, the standard library,
for general consumption of Rust programs. Adding a new one (libcore) is adding
more surface area to the distribution (in addition to adding a new `#![no_core]`
attribute). This surface area is greatly desired, however.

When using `#![no_std]` the experience of Rust programs isn't always the best as
there are some pitfalls that can be run into easily. For example, macros and
plugins sometimes hardcode `::std` paths, but most ones in the standard
distribution have been updated to use `::core` in the case that `#![no_std]` is
present. Another example is that common utilities like vectors, pointers, and
owned strings are not available without liballoc, which will remain an unstable
library. This means that users of `#![no_std]` will have to reimplement all of
this functionality themselves.

This RFC does not yet pave a way forward for using `#![no_std]` and producing an
executable because the `#[start]` item is required, but remains feature gated.
This RFC just enables creation of Rust static or dynamic libraries which don't
depend on the standard library in addition to Rust libraries (rlibs) which do
not depend on the standard library.

In stabilizing the `#![no_std]` attribute it's likely that a whole ecosystem of
crates will arise which work with `#![no_std]`, but in theory all of these
crates should also interoperate with the rest of the ecosystem using `std`.
Unfortunately, however, there are known cases where this is not possible. For
example if a macro is exported from a `#![no_std]` crate which references items
from `core` it won't work by default with a `std` library.

# Alternatives

Most of the strategies taken in this RFC have some minor variations on what can
happen:

* The `#![no_std]` attribute could be stabilized as-is without adding a
  `#![no_core]` attribute, requiring users to write `extern crate core` and
  import the core prelude manually. The burden of adding `#![no_core]` to the
  compiler, however, is seen as not-too-bad compared to the increase in
  ergonomics of using `#![no_std]`.
* Another stable crate could be provided by the distribution which provides
  definitions of these lang items which are all wired to abort. This has the
  downside of selecting a name for this crate, however, and also inflating the
  crates in our distribution again.

# Unresolved Questions

* How important/common are `#![no_std]` executables? Should this RFC attempt to
  stabilize that as well?
* When a staticlib is emitted should the compiler *guarantee* that a
  `#![no_std]` one will link by default? This precludes us from ever adding
  future require language items for features like unwinding or stack exhaustion
  by default. For example if a new security feature is added to LLVM and we'd
  like to enable it by default, it may require that a symbol or two is defined
  somewhere in the compilation.
