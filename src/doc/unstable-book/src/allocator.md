# `allocator`

The tracking issue for this feature is: [#27389]

[#27389]: https://github.com/rust-lang/rust/issues/27389

------------------------

Sometimes even the choices of jemalloc vs the system allocator aren't enough and
an entirely new custom allocator is required. In this you'll write your own
crate which implements the allocator API (e.g. the same as `alloc_system` or
`alloc_jemalloc`). As an example, let's take a look at a simplified and
annotated version of `alloc_system`

```rust,no_run
# // Only needed for rustdoc --test down below.
# #![feature(lang_items)]
// The compiler needs to be instructed that this crate is an allocator in order
// to realize that when this is linked in another allocator like jemalloc should
// not be linked in.
#![feature(allocator)]
#![allocator]

// Allocators are not allowed to depend on the standard library which in turn
// requires an allocator in order to avoid circular dependencies. This crate,
// however, can use all of libcore.
#![no_std]

// Let's give a unique name to our custom allocator:
#![crate_name = "my_allocator"]
#![crate_type = "rlib"]

// Our system allocator will use the in-tree libc crate for FFI bindings. Note
// that currently the external (crates.io) libc cannot be used because it links
// to the standard library (e.g. `#![no_std]` isn't stable yet), so that's why
// this specifically requires the in-tree version.
#![feature(libc)]
extern crate libc;

// Listed below are the five allocation functions currently required by custom
// allocators. Their signatures and symbol names are not currently typechecked
// by the compiler, but this is a future extension and are required to match
// what is found below.
//
// Note that the standard `malloc` and `realloc` functions do not provide a way
// to communicate alignment so this implementation would need to be improved
// with respect to alignment in that aspect.

#[no_mangle]
pub extern fn __rust_allocate(size: usize, _align: usize) -> *mut u8 {
    unsafe { libc::malloc(size as libc::size_t) as *mut u8 }
}

#[no_mangle]
pub extern fn __rust_deallocate(ptr: *mut u8, _old_size: usize, _align: usize) {
    unsafe { libc::free(ptr as *mut libc::c_void) }
}

#[no_mangle]
pub extern fn __rust_reallocate(ptr: *mut u8, _old_size: usize, size: usize,
                                _align: usize) -> *mut u8 {
    unsafe {
        libc::realloc(ptr as *mut libc::c_void, size as libc::size_t) as *mut u8
    }
}

#[no_mangle]
pub extern fn __rust_reallocate_inplace(_ptr: *mut u8, old_size: usize,
                                        _size: usize, _align: usize) -> usize {
    old_size // This api is not supported by libc.
}

#[no_mangle]
pub extern fn __rust_usable_size(size: usize, _align: usize) -> usize {
    size
}

# // Only needed to get rustdoc to test this:
# fn main() {}
# #[lang = "panic_fmt"] fn panic_fmt() {}
# #[lang = "eh_personality"] fn eh_personality() {}
# #[lang = "eh_unwind_resume"] extern fn eh_unwind_resume() {}
# #[no_mangle] pub extern fn rust_eh_register_frames () {}
# #[no_mangle] pub extern fn rust_eh_unregister_frames () {}
```

After we compile this crate, it can be used as follows:

```rust,ignore
extern crate my_allocator;

fn main() {
    let a = Box::new(8); // Allocates memory via our custom allocator crate.
    println!("{}", a);
}
```

## Custom allocator limitations

There are a few restrictions when working with custom allocators which may cause
compiler errors:

* Any one artifact may only be linked to at most one allocator. Binaries,
  dylibs, and staticlibs must link to exactly one allocator, and if none have
  been explicitly chosen the compiler will choose one. On the other hand rlibs
  do not need to link to an allocator (but still can).

* A consumer of an allocator is tagged with `#![needs_allocator]` (e.g. the
  `liballoc` crate currently) and an `#[allocator]` crate cannot transitively
  depend on a crate which needs an allocator (e.g. circular dependencies are not
  allowed). This basically means that allocators must restrict themselves to
  libcore currently.


