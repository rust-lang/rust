#![crate_type = "bin"]
#![no_main]
#![no_std]

extern crate a;
extern crate alloc;
extern crate b;

use alloc::vec::Vec;
use core::alloc::*;

struct Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        loop {}
    }

    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {
        loop {}
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[no_mangle]
extern "C" fn main(argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    let mut v = Vec::new();
    for i in 0..argc {
        v.push(i);
    }
    v.iter().sum()
}
