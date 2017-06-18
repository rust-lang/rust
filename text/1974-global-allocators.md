- Feature Name: `allocator`
- Start Date: 2017-02-04
- RFC PR: [rust-lang/rfcs#1974](https://github.com/rust-lang/rfcs/pull/1974)
- Rust Issue: [rust-lang/rust#27389](https://github.com/rust-lang/rust/issues/27389)

# Summary
[summary]: #summary

Overhaul the global allocator APIs to put them on a path to stabilization, and
switch the default allocator to the system allocator when the feature
stabilizes.

This RFC is a refinement of the previous [RFC 1183][].

[RFC 1183]: https://github.com/rust-lang/rfcs/blob/master/text/1183-swap-out-jemalloc.md

# Motivation
[motivation]: #motivation

## The current API

The unstable `allocator` feature allows developers to select the global
allocator which will be used in a program. A crate identifies itself as an
allocator with the `#![allocator]` annotation, and declares a number of
allocation functions with specific `#[no_mangle]` names and a C ABI. To
override the default global allocator, a crate simply pulls an allocator in
via an `extern crate`.

There are a couple of issues with the current approach:

A C-style ABI is error prone - nothing ensures that the signatures are correct,
and if a function is omitted that error will be caught by the linker rather than
compiler.

Allocators have some state, and with the current API, that state is forced to be
truly global since bare functions can't carry state.

Since an allocator is automatically selected when it is pulled into the crate
graph, it is painful to compose allocators. For example, one may want to create
an allocator which records statistics about active allocations, or adds padding
around allocations to attempt to detect buffer overflows in unsafe code. To do
this currently, the underlying allocator would need to be split into two
crates, one which contains all of the functionality and another which is tagged
as an `#![allocator]`.

## jemalloc

Rust's default allocator has historically been jemalloc. While jemalloc does
provide significant speedups over certain system allocators for some allocation
heavy workflows, it has has been a source of problems. For example, it has
deadlock issues on Windows, does not work with Valgrind, adds ~300KB to
binaries, and has caused crashes on macOS 10.12. See [this comment][] for more
details. As a result, it is already disabled on many targets, including all of
Windows. While there are certainly contexts in which jemalloc is a good choice,
developers should be making that decision, not the compiler. The system
allocator is a more reasonable and unsurprising default choice.

A third party crate allowing users to opt-into jemalloc would also open the door
to provide access to some of the library's other features such as tracing, arena
pinning, and diagnostic output dumps for code that depends on jemalloc directly.

[this comment]: https://github.com/rust-lang/rust/issues/36963#issuecomment-252029017

# Detailed design
[design]: #detailed-design

## Defining an allocator

Global allocators will use the `Allocator` trait defined in [RFC 1398][].
However `Allocator`'s methods take `&mut self` since it's designed to be used
with individual collections. Since this allocator is global across threads, we
can't take `&mut self` references to it. So, instead of implementing `Allocator`
for the allocator type itself, it is implemented for shared references to the
allocator. This is a bit strange, but similar to `File`'s `Read` and `Write`
implementations, for example.

```rust
pub struct Jemalloc;

impl<'a> Allocator for &'a Jemalloc {
    // ...
}
```

[RFC 1398]: https://github.com/rust-lang/rfcs/blob/master/text/1398-kinds-of-allocators.md

## Using an allocator

The `alloc::heap` module will contain several items:

```rust
/// Defined in RFC 1398
pub struct Layout { ... }

/// Defined in RFC 1398
pub unsafe trait Allocator { ... }

/// An `Allocator` which uses the system allocator.
///
/// This uses `malloc`/`free` on Unix systems, and `HeapAlloc`/`HeapFree` on
/// Windows, for example.
pub struct System;

unsafe impl Allocator for System { ... }

unsafe impl<'a> Allocator for &'a System { ... }

/// An `Allocator` which uses the configured global allocator.
///
/// The global allocator is selected by defining a static instance of the
/// allocator and annotating it with `#[global_allocator]`. Only one global
/// allocator can be defined in a crate graph.
///
/// # Note
///
/// For techical reasons, only non-generic methods of the `Allocator` trait
/// will be forwarded to the selected global allocator in the current
/// implementation.
pub struct Heap;

unsafe impl Allocator for Heap { ... }

unsafe impl<'a> Allocator for &'a Heap { ... }
```

This module will be reexported as `std::alloc`, which will be the location at
which it will be stabilized. The `alloc` crate is not proposed for stabilization
at this time.

An example of setting the global allocator:

```rust
extern crate my_allocator;

use my_allocator::{MyAllocator, MY_ALLOCATOR_INIT};

#[global_allocator]
static ALLOCATOR: MyAllocator = MY_ALLOCATOR_INIT;

fn main() {
    ...
}
```

Note that `ALLOCATOR` is still a normal static value - it can be used like any
other static would be.

The existing `alloc_system` and `alloc_jemalloc` crates will likely be
deprecated and eventually removed. The `alloc_system` crate is replaced with the
`SystemAllocator` structure in the standard library and the `alloc_jemalloc`
crate will become available on crates.io. The `alloc_jemalloc` crate will likely
look like:

```rust
pub struct Jemalloc;

unsafe impl Allocator for Jemalloc {
    // ...
}

unsafe impl<'a> Allocator for &'a Jemalloc {
    // ...
}
```

It is not proposed in this RFC to switch the per-platform default allocator just
yet. Assuming everything goes smoothly, however, it will likely be defined as
`System` as platforms transition away from jemalloc-by-default once the
jemalloc-from-crates.io is stable and usable.

The compiler will also no longer forbid cyclic the cyclic dependency between a
crate defining an implementation of an allocator and the `alloc` crate itself.
As a vestige of the current implementation this is only to get around linkage
errors where the liballoc rlib references symbols defined in the "allocator
crate". With this RFC the compiler has far more control over the ABI and linkage
here, so this restriction is no longer necessary.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Global allocator selection would be a somewhat advanced topic - the system
allocator is sufficient for most use cases. It is a new tool that developers can
use to optimize for their program's specific workload when necessary.

It should be emphasized that in most cases, the "terminal" crate (i.e. the bin,
cdylib or staticlib crate) should be the only thing selecting the global
allocator. Libraries should be agnostic over the global allocator unless they
are specifically designed to augment functionality of a specific allocator.

Defining an allocator is an even more advanced topic that should probably live
in the _Nomicon_.

[RFC 1398]: https://github.com/rust-lang/rfcs/pull/1398

# Drawbacks
[drawbacks]: #drawbacks

Dropping the default of jemalloc will regress performance of some programs until
they manually opt back into that allocator, which may produce confusion in the
community as to why things suddenly became slower.

Depending on implementation of a trait for references to a type is unfortunate.
It's pretty strange and unfamiliar to many Rust developers. Many global
allocators are zero-sized as their state lives outside of the Rust structure,
but a reference to the allocator will be 4 or 8 bytes. If developers wish to use
global allocators as "normal" allocators in individual collections, allocator
authors may have to implement `Allocator` twice - for the type and references to
the type. One can forward to the other, but it's still work that would not need
to be done ideally.

In theory, there could be a blanket implementation of `impl<'a, T> Allocator for
T where &'a T: Allocator`, but the compiler is unfortunately not able to deal
with this currently.

The `Allocator` trait defines some functions which have generic arguments.
They're purely convenience functions, but if a global allocator overrides them,
the custom implementations will not be used when going through the `Heap` type.
This may be confusing.

# Alternatives
[alternatives]: #alternatives

We could define a separate `GlobalAllocator` trait with methods taking `&self`
to avoid the strange implementation for references requirement. This does
require the duplication of some or all of the API surface and documentation of
`Allocator` to a second trait with only a difference in receiver type.

The `GlobalAllocator` trait could be responsible for simply returning a type
which implements `Allocator`. This avoids the duplication or the strange
implementation for references issues in the other possibilities, but can't be
defined in a reasonable way without HKT, and is a somewhat strange layer of
indirection.

# Unresolved questions
[unresolved]: #unresolved-questions

Are `System` and `Heap` the right names for the two `Allocator` implementations
in `std::heap`?

Should `std::heap` also have free functions which forward to the global
allocator?
