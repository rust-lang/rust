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
compiler. The Macros 1.1 API is similar in that certain special functions must
be identified to the compiler, and in that case a special attribute
(`#[proc_macro_derive]`)is used rather than a magic symbol name.

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

An allocator crate identifies itself as such by applying the `#![allocator]`
annotate at the crate root. It then defines a specific set of functions which
are tagged with attributes:

```rust
#![allocator]

/// Returns a pointer to `size` bytes of memory aligned to `align`.
///
/// On failure, returns a null pointer.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
#[allocator(allocate)]
pub fn allocate(size: usize, align: usize) -> *mut u8 {
    ...
}

/// Returns a pointer to `size` bytes of memory aligned to `align`, and
/// initialized with zeroes.
///
/// On failure, returns a null pointer.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
#[allocator(allocate_zeroed)]
pub fn allocate_zeroed(size: usize, align: usize) -> *mut u8 {
    ...
}

/// Deallocates the memory referenced by `ptr`.
///
/// The `ptr` parameter must not be null.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`.
#[allocator(deallocate)]
pub fn deallocate(ptr: *mut u8, old_size: usize, align: usize) {
    ...
}

/// Resizes the allocation referenced by `ptr` to `size` bytes.
///
/// On failure, returns a null pointer and leaves the original allocation
/// intact.
///
/// If the allocation was relocated, the memory at the passed-in pointer is
/// undefined after the call.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`.
#[allocator(reallocate)]
pub fn reallocate(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> *mut u8 {
    ...
}

/// Resizes the allocation referenced by `ptr` to `size` bytes without moving
/// it.
///
/// The new size of the allocation is returned. This must be at least
/// `old_size`. The allocation must always remain valid.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`.
///
/// This function is optional. The default implementation simply returns
/// `old_size`.
#[allocator(reallocate_inplace)]
pub fn reallocate_inplace(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> usize {
    ...
}
```

Note that `useable_size` has been removed, as it is not used anywhere in the
standard library.

The allocator functions must be publicly accessible, but can have any name and
be defined in any module. However, it is recommended to use the names above in
the crate root to minimize confusion.

An allocator must provide all functions with the exception of
`reallocate_inplace`. New functions can be added to the API in the future in a
similar way to `reallocate_inplace`.

## Using an allocator

The functions that an allocator crate defines can be called directly, but most
usage will happen through the *global allocator* interface located in
`std::heap`. This module exposes a set of functions identical to those described
above, but that call into the global allocator. To select the global allocator,
a crate declares it via an `extern crate` annotated with `#[allocator]`:

```rust
#[allocator]
extern crate jemalloc;
```

As its name would suggest, the global allocator is a global resource - all
crates in a dependency tree must agree on the selected global allocator. If two
or more distinct allocator crates are selected, compilation will fail. Note that
multiple crates can select a global allocator as long as that allocator is the
same across all of them. In addition, a crate can depend on an allocator crate
without declaring it to be the global allocator by omitting the `#[allocator]`
annotation.

## Standard library

The standard library will gain a new stable crate - `alloc_system`. This is the
default allocator crate and corresponds to the "system" allocator (i.e. `malloc`
etc on Unix and `HeapAlloc` etc on Windows).

The `alloc::heap` module will be reexported in `std` and stabilized. It will
simply contain functions matching directly to those defined by the allocator
API. The `alloc` crate itself may also be stabilized at a later date, but this
RFC does not propose that.

The existing `alloc_jemalloc` may continue to exist as an implementation detail
of the Rust compiler, but it will never be stabilized. Applications wishing to
use jemalloc can use a third-party crate from crates.io.

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

We could require that at most one crate selects a global allocator in the crate
graph, which may simplify the implementation.

The allocator APIs could be simplified to a more "traditional"
malloc/calloc/free API at the cost of an efficiency loss when using allocators
with more powerful APIs.

# Unresolved questions
[unresolved]: #unresolved-questions

It is currently forbidden to pass a null pointer to `deallocate`, though this is
guaranteed to be a noop with libc's `free` at least. Some kinds of patterns in C
are cleaner when null pointers can be `free`d - is the same true for Rust?
