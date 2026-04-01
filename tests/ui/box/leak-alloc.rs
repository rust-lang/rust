#![feature(allocator_api)]

use std::alloc::{AllocError, Allocator, Layout, System};
use std::ptr::NonNull;

use std::boxed::Box;

struct Alloc {}

unsafe impl Allocator for Alloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        System.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        System.deallocate(ptr, layout)
    }
}

fn use_value(_: u32) {}

fn main() {
    let alloc = Alloc {};
    let boxed = Box::new_in(10, alloc.by_ref());
    let theref = Box::leak(boxed);
    drop(alloc);
    //~^ ERROR cannot move out of `alloc` because it is borrowed
    use_value(*theref)
}
