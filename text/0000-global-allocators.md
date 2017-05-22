- Feature Name: `allocator`
- Start Date: 2017-02-04
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Overhaul the global allocator APIs to put them on a path to stabilization, and
switch the default allocator to the system allocator when the feature
stabilizes.

This RFC is a refinement of the previous [RFC 1183][].

[RFC 1183]: https://github.com/rust-lang/rfcs/blob/master/text/1183-swap-out-jemalloc.md

# Motivation
[motivation]: #motivation

## API

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

We introduce a new trait, `GlobalAllocator`. It is similar to the `Allocator`
trait described in [RFC 1398][], but is stripped down and the methods take
`&self` rather than `&mut self`.

[RFC 1398]: https://github.com/rust-lang/rfcs/blob/master/text/1398-kinds-of-allocators.md

```rust
/// A trait implemented by objects that can be global allocators.
///
/// Instances of this trait can be used to back allocations done through the
/// `std::heap` API. This trait represents the fundamental ability to allocate
/// memory in Rust.
///
/// To use a global allocator you'll need to use the `#[global_allocator]`
/// attribute like so:
///
/// ```
/// extern crate my_allocator;
///
/// #[global_allocator]
/// static ALLOCATOR: MyAllocator = my_allocator::INIT;
///
/// fn main() {
///     let _b = Box::new(2); // uses `MyAllocator` above
/// }
/// ```
///
/// # Unsafety
///
/// This trait is an `unsafe` trait as there are a number of guarantees a global
/// allocator must adhere to which aren't expressible through the type system.
/// First and foremost types that implement this trait must behave like, well,
/// allocators! All pointers returned from `allocate` that are active in a
/// program (disregarding those `deallocate`d) must point to disjoint chunks of
/// memory. In other words, allocations need to be distinct and can't overlap.
///
/// Additionally it must be safe to allocate a chunk of memory on any thread of
/// a program and then deallocate it on any other thread of the program.
#[lang = "global_allocator"]
pub unsafe trait GlobalAllocator: Send + Sync + 'static {
    /// Returns a pointer to a newly allocated region of memory suitable for the
    /// provided `Layout`. The contents of the memory are undefined.
    ///
    /// On failure, returns a null pointer.
    pub fn allocate(&self, layout: Layout) -> *mut u8;

    /// Returns a pointer to a newly allocated region of memory suitable for the
    /// provided `Layout`. The memory is guaranteed to contain zeroes.
    ///
    /// On failure, returns a null pointer.
    pub fn allocate_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.allocate(layout);
        if !ptr.is_null() {
            ptr::write_bytes(ptr, 0, layout.size());
        }
        ptr
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// The pointer must correspond to a region of memory previously allocated
    /// by this allocator with the provided layout.
    pub unsafe fn deallocate(&self, ptr: *mut u8, layout: Layout);

    /// Resizes the allocation referenced by `ptr` a new layout.
    ///
    /// On failure, returns a null pointer and leaves the original allocation
    /// intact.
    ///
    /// If the allocation was relocated, the memory at the passed-in pointer is
    /// undefined after the call.
    ///
    /// The pointer must correspond to a region of memory previously allocated
    /// by this allocator with the provided layout.
    pub unsafe fn reallocate(&self, ptr: *mut u8, old_layout: Layout, layout: Layout) -> *mut u8 {
        let new_ptr = self.alloc(layout);
        if !new_ptr.is_null() {
            ptr::copy_nonoverlapping(ptr, new_ptr, cmp::min(old_layout.size(), layout.size()));
            self.deallocate(ptr);
        }
        new_ptr
    }
}
```

Two methods currently defined in the global allocatr API are not present on this
trait: `usable_size` which is used nowhere in the standard library, and
`reallocate_inplace`, which is only used in libarena.

A global allocator is a type implementing `GlobalAllocator` which can be
constructed in a constant expression.

Note that the precise type signatures used here are a little up for debate. It's
expected that they will be settled (along with accompanying documentation as to
guarantees) as part of stabilization in [rust-lang/rust#27700][stab-issue].

[stab-issue]: https://github.com/rust-lang/rust/issues/27700

## Using an allocator

While the `GlobalAllocator` trait can be used like any other, the most common
usage of a global allocator is through the functions defined in the
`std::heap` module. It contains free functions corresponding to each of the
methods defined on the `GlobalAllocator` trait:

```rust
pub fn allocate(layout: Layout) -> *mut u8 {
    ...
}

pub fn allocate_zeroed(layout: Layout) -> *mut u8 {
    ...
}

pub unsafe fn deallocate(&self, ptr: *mut u8, layout: Layout) {
    ...
}

pub unsafe fn reallocate(ptr: *mut u8, old_layout: Layout, layout: Layout) -> *mut u8 {
    ...
}
```

Each of these functions simply delegates to the selected global allocator. The
allocator is selected by tagging a static value of a type implementing
`GlobalAllocator` with the `#[global_allocator]` annotation:

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
other static would bed.

## Standard library

A `core::heap` module will be added with the `Layout` type and the
`GlobalAllocator` trait. The initial API of `Layout` will be more conservative
than that described in [RFC 1398][], possibly nothing more than a
`from_size_align` constructor and accessors for `size` and `align`. It is
intended that the API will grow over time with conveniences such as
`Layout::new::<T>()`.

The `alloc::heap` module will reexport these types from `core::heap`. It will
also provide top-level functions (like `allocate` above) which do not take an
allocator but instead are routed through the global allocator. The `alloc`
crate, however, will not provide a global allocator. Instead the compiler will
understand that crates which transitively depend on `alloc` will require an
allocator, with a per-target fallback default allocator used by the compiler.

The standard library will grow a `std::heap` module that reexports the contents
of `alloc::heap`. Additionally it will contain a system allocator definition:

```rust
pub struct SystemAllocator;

impl GlobalAllocator for SystemAllocator {
    // ...
}
```

The `SystemAllocator` is defined as having zero size, no fields, and always
referring to the OS allocator (i.e. `malloc` etc on Unix and `HeapAlloc` etc on
Windows).

The existing `alloc_system` and `alloc_jemalloc` crates will likely be
deprecated and eventually removed. The `alloc_system` crate is replaced with the
`SystemAllocator` structure in the standard library and the `alloc_jemalloc`
crate will become available on crates.io. The `alloc_jemalloc` crate will likely
look like:

```rust
pub struct Jemalloc;

impl GlobalAllocator for Jemalloc {
    // ...
}
```

It is not proposed in this RFC to switch the per-platform default allocator just
yet. Assuming everything goes smoothly, however, it will likely be defined as
`SystemAllocator` as platforms transition away from jemalloc-by-default once the
jemalloc-from-crates.io is stable and usable.

The compiler will also no longer forbid cyclic the cyclic dependency between a
crate defining an implementation of an allocator and the `alloc` crate itself.
As a vestige of the current implementation this is only to get around linkage
errors where the liballoc rlib references symbols defined in the "allocator
crate". With this RFC the compiler has far more control over the ABI and linkage
here, so this restriction is no longer necessary.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The term "allocator" is the canonical one for this concept. It is unfortunately
shared with a similar but distinct concept described in [RFC 1398][], which
defined an `Allocator` trait over which collections be parameterized. This API
is disambiguated by referring specifically to the "global" or "default"
allocator.

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

The allocator APIs are to some extent designed after what jemalloc supports,
which is quite a bit more than the system allocator is able to. The Rust
wrappers for those simpler allocators have to jump through hoops to ensure that
all of the requirements are met.

# Alternatives
[alternatives]: #alternatives

We could loosen the requirement that the root crate is the only one which may
select the global allocator in favor of allowing any crate in the dependency
graph to do so.

We could try to use the `Allocator` trait for global allocators. The `&mut self`
problem can b e solved via an implementation on a reference to the allocator
type in a way similar to `TcpStream`'s `Write` and `Read` implementations, but
this is pretty hacky.

# Unresolved questions
[unresolved]: #unresolved-questions

It is currently forbidden to pass a null pointer to `deallocate`, though this is
guaranteed to be a noop with libc's `free` at least. Some kinds of patterns in C
are cleaner when null pointers can be `free`d - is the same true for Rust?

The `Allocator` trait defines several methods that do not have corresponding
implementations here:

* `oom`, which is called after a failed allocation to provide any allocator
  specific messaging that may exist.
* `usable_size`, which is mentioned above as being unused, and should probably
  be removed from this trait as well.
* `realloc_inplace`, which attempts to resize an allocation without moving it.
* `alloc_excess`, which is like `alloc` but returns the entire usable size
  including any extra space beyond the requested size.
* Some other higher level convenience methods like `alloc_array`.

Should any of these be added to the global allocator as well? It may make sense
to add `alloc_excess` to the allocator API. This can either have a default
implementation which simply calls `allocate` and returns the input size, or
calls `allocate` followed by `reallocate_inplace`.

The existing `usable_size` function (proposed for removal) only takes a size and
align. A similar, but potentially more useful API is one that takes a pointer
to a heap allocated region and returns the usable size of it. This is supported
as a GNU extension `malloc_useable_size` in the system allocator, and in
jemalloc as well. An [issue][usable_size] has been filed to add this to support
this to aid heap usage diagnostic crates. It would presumably have to return an
`Option` to for allocators that do not have such a function, but this limits
its usefulness if support can't be guaranteed.

[usable_size]: https://github.com/rust-lang/rust/issues/32075
