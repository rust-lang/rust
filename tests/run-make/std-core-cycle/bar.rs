#![feature(allocator_api)]
#![crate_type = "rlib"]

use std::alloc::*;

pub struct A;

unsafe impl GlobalAlloc for A {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        loop {}
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _: Layout) {
        loop {}
    }
}
