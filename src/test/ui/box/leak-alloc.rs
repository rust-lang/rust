#![feature(allocator_api)]

use std::alloc::{AllocError, AllocRef, Layout, System};
use std::ptr::NonNull;

use std::boxed::Box;

struct Allocator {}

unsafe impl AllocRef for Allocator {
    fn alloc(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

fn use_value(_: u32) {}

fn main() {
    let alloc = Allocator {};
    let boxed = Box::new_in(10, alloc.by_ref());
    let theref = Box::leak(boxed);
    drop(alloc);
    //~^ ERROR cannot move out of `alloc` because it is borrowed
    use_value(*theref)
}
