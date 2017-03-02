- Feature Name: `allocator`
- Start Date: 2015-06-27
- RFC PR: [rust-lang/rfcs#1183](https://github.com/rust-lang/rfcs/pull/1183)
- Rust Issue: [rust-lang/rust#27389](https://github.com/rust-lang/rust/issues/27389)

# Summary

Add support to the compiler to override the default allocator, allowing a
different allocator to be used by default in Rust programs. Additionally, also
switch the default allocator for dynamic libraries and static libraries to using
the system malloc instead of jemalloc.

# Motivation

Note that this issue was [discussed quite a bit][babysteps] in the past, and
the meat of this RFC draws from Niko's post.

[babysteps]: http://smallcultfollowing.com/babysteps/blog/2014/11/14/allocators-in-rust/

Currently all Rust programs by default use jemalloc for an allocator because it
is a fairly reasonable default as it is commonly much faster than the default
system allocator. This is not desirable, however, when embedding Rust code into
other runtimes. Using jemalloc implies that Rust will be using one allocator
while the host application (e.g. Ruby, Firefox, etc) will be using a separate
allocator. Having two allocators in one process generally hurts performance and
is not recommended, so the Rust toolchain needs to provide a method to configure
the allocator.

In addition to using an entirely separate allocator altogether, some Rust
programs may want to simply instrument allocations or shim in additional
functionality (such as memory tracking statistics). This is currently quite
difficult to do, and would be accomodated with a custom allocation scheme.

# Detailed design

The high level design can be found [in this gist][gist], but this RFC intends to
expound on the idea to make it more concrete in terms of what the compiler
implementation will look like. A [sample implementaiton][impl] is available of
this section.

[gist]: https://gist.github.com/alexcrichton/41c6aad500e56f49abda
[impl]: https://github.com/alexcrichton/rust/tree/less-jemalloc

### High level design

The design of this RFC from 10,000 feet (referred to below), which was
[previously outlined][gist] looks like:

1. Define a set of symbols which correspond to the APIs specified in
   `alloc::heap`. The `liballoc` library will call these symbols directly.
   Note that this means that each of the symbols take information like the size
   of allocations and such.
2. Create two shim libraries which implement these allocation-related functions.
   Each shim is shipped with the compiler in the form of a static library. One
   shim will redirect to the system allocator, the other shim will bundle a
   jemalloc build along with Rust shims to redirect to jemalloc.
3. Intermediate artifacts (rlibs) do not resolve this dependency, they're just
   left dangling.
4. When producing a "final artifact", rustc by default links in one of two
   shims:
    * If we're producing a staticlib or a dylib, link the system shim.
    * If we're producing an exe and all dependencies are rlibs link the
      jemalloc shim.

The final link step will be optional, and one could link in any compliant
allocator at that time if so desired.

### New Attributes

Two new **unstable** attributes will be added to the compiler:

* `#![needs_allocator]` indicates that a library requires the "allocation
  symbols" to link successfully. This attribute will be attached to `liballoc`
  and no other library should need to be tagged as such. Additionally, most
  crates don't need to worry about this attribute as they'll transitively link
  to liballoc.
* `#![allocator]` indicates that a crate is an allocator crate. This is
  currently also used for tagging FFI functions as an "allocation function"
  to leverage more LLVM optimizations as well.

All crates implementing the Rust allocation API must be tagged with
`#![allocator]` to get properly recognized and handled.

### New Crates

Two new **unstable** crates will be added to the standard distribution:

* `alloc_system` is a crate that will be tagged with `#![allocator]` and will
  redirect allocation requests to the system allocator.
* `alloc_jemalloc` is another allocator crate that will bundle a static copy of
  jemalloc to redirect allocations to.

Both crates will be available to link to manually, but they will not be
available in stable Rust to start out.

### Allocation functions

Each crate tagged `#![allocator]` is expected to provide the full suite of
allocation functions used by Rust, defined as:

```rust
extern {
    fn __rust_allocate(size: usize, align: usize) -> *mut u8;
    fn __rust_deallocate(ptr: *mut u8, old_size: usize, align: usize);
    fn __rust_reallocate(ptr: *mut u8, old_size: usize, size: usize,
                         align: usize) -> *mut u8;
    fn __rust_reallocate_inplace(ptr: *mut u8, old_size: usize, size: usize,
                                 align: usize) -> usize;
    fn __rust_usable_size(size: usize, align: usize) -> usize;
}
```

The exact API of all these symbols is considered **unstable** (hence the
leading `__`). This otherwise currently maps to what `liballoc` expects today.
The compiler will not currently typecheck `#![allocator]` crates to ensure
these symbols are defined and have the correct signature.

Also note that to define the above API in a Rust crate it would look something
like:

```rust
#[no_mangle]
pub extern fn __rust_allocate(size: usize, align: usize) -> *mut u8 {
    /* ... */
}
```

### Limitations of `#![allocator]`

Allocator crates (those tagged with `#![allocator]`) are not allowed to
transitively depend on a crate which is tagged with `#![needs_allocator]`. This
would introduce a circular dependency which is difficult to link and is highly
likely to otherwise just lead to infinite recursion.

The compiler will also not immediately verify that crates tagged with
`#![allocator]` do indeed define an appropriate allocation API, and vice versa
if a crate defines an allocation API the compiler will not verify that it is
tagged with `#![allocator]`. This means that the only meaning `#![allocator]`
has to the compiler is to signal that the default allocator should not be
linked.

### Default allocator specifications

Target specifications will be extended with two keys: `lib_allocation_crate`
and `exe_allocation_crate`, describing the default allocator crate for these
two kinds of artifacts for each target. The compiler will by default have all
targets redirect to `alloc_system` for both scenarios, but `alloc_jemalloc` will
be used for binaries on OSX, Bitrig, DragonFly, FreeBSD, Linux, OpenBSD, and GNU
Windows. MSVC will notably **not** use jemalloc by default for binaries (we
don't currently build jemalloc on MSVC).

### Injecting an allocator

As described above, the compiler will inject an allocator if necessary into the
current compilation. The compiler, however, cannot blindly do so as it can
easily lead to link errors (or worse, two allocators), so it will have some
heuristics for only injecting an allocator when necessary. The steps taken by
the compiler for any particular compilation will be:

* If no crate in the dependency graph is tagged with `#![needs_allocator]`, then
  the compiler does not inject an allocator.
* If only an rlib is being produced, no allocator is injected.
* If any crate tagged with `#[allocator]` has been explicitly linked to (e.g.
  via an `extern crate` statement directly or transitively) then no allocator is
  injected.
* If two allocators have been linked to explicitly an error is generated.
* If only a binary is being produced, then the target's `exe_allocation_crate`
  value is injected, otherwise the `lib_allocation_crate` is injected.

The compiler will also record that the injected crate is injected, so later
compilations know that rlibs don't actually require the injected crate at
runtime (allowing it to be overridden).

### Allocators in practice

Most libraries written in Rust wouldn't interact with the scheme proposed in
this RFC at all as they wouldn't explicitly link with an allocator and generally
are compiled as rlibs. If a Rust dynamic library is used as a dependency, then
its original choice of allocator is propagated throughout the crate graph, but
this rarely happens (except for the compiler itself, which will continue to use
jemalloc).

Authors of crates which are embedded into other runtimes will start using the
system allocator by default with no extra annotation needed. If they wish to
funnel Rust allocations to the same source as the host application's allocations
then a crate can be written and linked in.

Finally, providers of allocators will simply provide a crate to do so, and then
applications and/or libraries can make explicit use of the allocator by
depending on it as usual.

# Drawbacks

A significant amount of API surface area is being added to the compiler and
standard distribution as part of this RFC, but it is possible for it to all
enter as `#[unstable]`, so we can take our time stabilizing it and perhaps only
stabilize a subset over time.

The limitation of an allocator crate not being able to link to the standard
library (or libcollections) may be a somewhat significant hit to the ergonomics
of defining an allocator, but allocators are traditionally a very niche class of
library and end up defining their own data structures regardless.

Libraries on crates.io may accidentally link to an allocator and not actually
use any specific API from it (other than the standard allocation symbols),
forcing transitive dependants to silently use that allocator.

This RFC does not specify the ability to swap out the allocator via the command
line, which is certainly possible and sometimes more convenient than modifying
the source itself.

It's possible to define an allocator API (e.g. define the symbols) but then
forget the `#![allocator]` annotation, causing the compiler to wind up linking
two allocators, which may cause link errors that are difficult to debug.

# Alternatives

The compiler's knowledge about allocators could be simplified quite a bit to the
point where a compiler flag is used to just turn injection on/off, and then it's
the responsibility of the application to define the necessary symbols if the
flag is turned off. The current implementation of this RFC, however, is not seen
as overly invasive and the benefits of "everything's just a crate" seems worth
it for the mild amount of complexity in the compiler.

Many of the names (such as `alloc_system`) have a number of alternatives, and
the naming of attributes and functions could perhaps follow a stronger
convention.

# Unresolved questions

Does this enable jemalloc to be built without a prefix on Linux? This would
enable us to direct LLVM allocations to jemalloc, which would be quite nice!

Should BSD-like systems use Rust's jemalloc by default? Many of them have
jemalloc as the system allocator and even the special APIs we use from jemalloc.
