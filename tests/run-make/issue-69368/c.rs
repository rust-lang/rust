#![crate_type = "bin"]
#![feature(start)]
#![no_std]

extern crate alloc;
extern crate a;
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

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
    let mut v = Vec::new();
    for i in 0..argc {
        v.push(i);
    }
    v.iter().sum()
}
