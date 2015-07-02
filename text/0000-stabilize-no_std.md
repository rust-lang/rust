- Feature Name: N/A
- Start Date: 2015-06-26
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Stabilize the `#![no_std]` attribute, add a new `#![no_core]` attribute, and
start stabilizing the libcore library.

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

* Stabilize the `#![no_std]` attribute after tweaking its behavior slightly
* Introduce a `#![no_core]` attribute.
* Stabilize the name "core" in libcore.
* Stabilize required language items by the core library.

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
but it proposes stabilizing the name `core` for libcore, paving the way for the
rest of the library to be stabilized. The exact method of stabilizing its
contents will be determined with a future RFC or pull requests.

## Stabilizing lang items

This section will describe the purpose for each lang item currently required in
addition to the interface that it will be stabilized with. Each lang item will
no longer be defined with the `#[lang = "..."]` syntax but will instead receive
a dedicated attribute (e.g. `#[panic_fmt]`) to be attached to functions to
identify an implementation. It should be noted that these language items are
already not quite the same as other `#[lang]` items due to the ability to rely
on them in a "weak" fashion.

Like lang items each of these will only allow one implementor in any crate
dependency graph which will be verified at compile time. Also like today, none
of these lang items will be required unless a static library, dynamic library,
or executable is being produced. In other words, libraries (rlibs) do not need
(and probably should not) to define these items.

#### `panic_fmt`

This lang item is the definition of how to panic in Rust. The standard library
defines this by throwing an exception (in a platform-specific manner), but users
of libcore often want to define their own meaning of panicking. The signature of
this function will be:

```rust
#[panic_fmt]
pub extern fn panic_fmt(msg: &core::fmt::Arguments) -> !;
```

This differs with the `panic_fmt` function today in that the file and line
number arguments are omitted. The libcore library will continue to provide
file/line number information in panics (as it does today) by assembling a new
`core::fmt::Arguments` value which uses the old one and appends the file/line
information.

This signature also differs from today's implementation by taking a `&Arguments`
instead of taking it by value, and the purpose of this is to ensure that the
function has a clearly defined ABI on all platforms in case that is required.

#### `eh_personality`

The compiler will continue to compile libcore with landing pads (e.g. cleanup to
run on panics), and a "personality function" is required by LLVM to be available
to call for each landing pad. In the current implementation of panicking, a
personality function is typically just calling a standard personality function
in libgcc (or in MSVC's CRT), but the purpose is to indicate whether an
exception should be caught or whether cleanup should be run for this particular
landing pad and exception combination.

The exact signature of this function is quite platform-specific, but many users
of libcore will never actually call this function as exceptions will not be
thrown (many will likely compile with `-Z no-landing-pads` anyway). As a result
the signature of this lang item will not be defined, but instead it will simply
be required to be defined (as libcore will reference the symbol name
regardless).

```rust
#[eh_personality]
pub extern fn eh_personality(...) -> ...;
```

The compiler will not check the signature of this function, but it will assign
it a known symbol so libcore can be successfully linked.

#### `stack_exhausted`

The current implementation of stack overflow in the compiler is to use LLVM's
segmented stack support, inserting a prologue to every function in an object
file to detect when a stack overflow occurred. When a stack overflow is
detected, LLVM emits code that will call the symbol `__morestack`, which the
Rust distribution provides an implementation of. Our implementation, however,
then in turn calls a this `stack_exhausted` language item to define the
implementation of what happens on stack overflow.

The compiler therefore needs to ensure that this lang item is present in order
for libcore to be correctly linked, so the lang item will have the following
signature:

```rust
#[stack_exhausted]
pub extern fn stack_exhausted() -> !;
```

The compiler will control the symbol name and visibility of this function.

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

On the topic of lang item stabilization, it's likely expected that the
`panic_fmt` lang item must be defined, but the other two, `eh_personality` and
`stack_exhausted` are generally quite surprising. Code using `#![no_std]` is
also likely to very rarely actually make use of these functions:

* Most no-std contexts don't throw exceptions (or don't have exceptions), so
  they either have stubs that panic or just compile with `-Z no-landing-pads`,
  so the `eh_personality` may not strictly be necessary to be defined in order
  to link against libcore.
* Additionally, most no-std contexts don't actually set up stack overflow
  detection, so the `stack_exhausted` function will either never be compiled or
  the crates are compiled with `-C no-stack-check` meaning that the item may not
  strictly be necessary to be defined.

Currently, however, a binary distribution of libcore is provided which is
compiled with unwinding and stack overflow checks enabled. Consequently the
libcore library does indeed depend on these two symbols and require these items
to be defined. It is seen as not-that-large of a drawback for the following
reasons:

* The functions `eh_personality` and `stack_exhausted` are fairly easy to
  define, and are only required by end products (not Rust libraries).
* It's easy for the compiler to *stop* requiring these functions to be defined
  in the future if we, for example, provide multiple binary copies of libcore in
  the standard distribution.

Another drawback of this RFC is the overall stabilization of the `#![no_std]`
attribute, meaning that the compiler will no longer be able to make assumptions
in the future about a function being defined. Put another way, the `panic_fmt`,
`eh_personality`, and `stack_exhausted` lang items are the only three that will
ever be able to be required to be defined by downstream crates. This is not seen
as too strong of a drawback as it's not clear that the compiler will need to
assume more functions exist. Additionally, the compiler will likely be able to
provide or emit a stub implementation for any future symbol it does need to
exist.

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
* The language items could continue to use the same `#[lang = "..."]` syntax and
  we could just stabilize a subset of the `#[lang]` items. It seems more
  consistent, however, to blanket feature-gate all `#[lang]` attributes instead
  of allowing three particular ones, so individual attributes are proposed.
* The `panic_fmt` lang item could retain the same signature today, but it has an
  unclear ABI (passing `Arguments` by value) and we may not want to 100% commit
  to always passing filename/line information on panics.
* The `eh_personality` and `stack_exhausted` lang items could not be required to
  be defined, and the compiler could provide aborting stubs to be linked in if
  they aren't defined anywhere else.
* The compiler could not require `eh_personality` or `stack_exhausted` if no
  crate in the dependency tree has landing pads enabled or stack overflow checks
  enabled. This is quite a difficult situation to get into today, however, as
  the libcore distribution always has these enabled and Cargo does not easily
  provide a method to configure this when compiling crates. The overhead of
  defining these functions seems small and because the compiler could stop
  requiring them in the future it seems plausibly ok to require them today.
* A `#[lang_items_abort]` attribute could be added to explicitly define the the
  `eh_personality` and `stack_exhausted` lang items to immediately abort. This
  would avoid us having to stabilize their signatures as we could stabilize just
  this attribute and not their definitions.
* The various language items could not be stabilized at this time, allowing
  stable libraries that leverage `#![no_std]` but not stable final artifacts
  (e.g. staticlibs, dylibs, or binaries).

# Unresolved Questions

* How important/common are `#![no_std]` executables? Should this RFC attempt to
  stabilize that as well?
* When a staticlib is emitted should the compiler *guarantee* that a
  `#![no_std]` one will link by default? This precludes us from ever adding
  future require language items for features like unwinding or stack exhaustion
  by default. For example if a new security feature is added to LLVM and we'd
  like to enable it by default, it may require that a symbol or two is defined
  somewhere in the compilation.
