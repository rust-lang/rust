//@ compile-flags: --crate-type cdylib -C lto
//@ build-pass
//@ no-prefer-dynamic
//@ needs-crate-type: cdylib

use std::alloc::{GlobalAlloc, Layout};

struct MyAllocator;

unsafe impl GlobalAlloc for MyAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        todo!()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static GLOBAL: MyAllocator = MyAllocator;
