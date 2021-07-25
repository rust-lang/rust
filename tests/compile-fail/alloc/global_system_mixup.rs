// Make sure we detect when the `Global` and `System` allocators are mixed
// (even when the default `Global` uses `System`).
// error-pattern: which is Rust heap memory, using

#![feature(allocator_api, slice_ptr_get)]

use std::alloc::{Allocator, Global, System, Layout};

fn main() {
    let l = Layout::from_size_align(1, 1).unwrap();
    let ptr = Global.allocate(l).unwrap().as_non_null_ptr();
    unsafe { System.deallocate(ptr, l); }
}
