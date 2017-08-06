# `global_allocator`

The tracking issue for this feature is: [#27389]

[#27389]: https://github.com/rust-lang/rust/issues/27389

------------------------

Rust programs may need to change the allocator that they're running with from
time to time. This use case is distinct from an allocator-per-collection (e.g. a
`Vec` with a custom allocator) and instead is more related to changing the
global default allocator, e.g. what `Vec<T>` uses by default.

Currently Rust programs don't have a specified global allocator. The compiler
may link to a version of [jemalloc] on some platforms, but this is not
guaranteed. Libraries, however, like cdylibs and staticlibs are guaranteed
to use the "system allocator" which means something like `malloc` on Unixes and
`HeapAlloc` on Windows.

[jemalloc]: https://github.com/jemalloc/jemalloc

The `#[global_allocator]` attribute, however, allows configuring this choice.
You can use this to implement a completely custom global allocator to route all
default allocation requests to a custom object. Defined in [RFC 1974] usage
looks like:

[RFC 1974]: https://github.com/rust-lang/rfcs/pull/1974

```rust
#![feature(global_allocator, allocator_api, heap_api)]

use std::heap::{Alloc, System, Layout, AllocErr};

struct MyAllocator;

unsafe impl<'a> Alloc for &'a MyAllocator {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        System.alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: MyAllocator = MyAllocator;

fn main() {
    // This `Vec` will allocate memory through `GLOBAL` above
    let mut v = Vec::new();
    v.push(1);
}
```

And that's it! The `#[global_allocator]` attribute is applied to a `static`
which implements the `Alloc` trait in the `std::heap` module. Note, though,
that the implementation is defined for `&MyAllocator`, not just `MyAllocator`.
You may wish, however, to also provide `Alloc for MyAllocator` for other use
cases.

A crate can only have one instance of `#[global_allocator]` and this instance
may be loaded through a dependency. For example `#[global_allocator]` above
could have been placed in one of the dependencies loaded through `extern crate`.

Note that `Alloc` itself is an `unsafe` trait, with much documentation on the
trait itself about usage and for implementors. Extra care should be taken when
implementing a global allocator as well as the allocator may be called from many
portions of the standard library, such as the panicking routine. As a result it
is highly recommended to not panic during allocation and work in as many
situations with as few dependencies as possible as well.
