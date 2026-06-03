#![feature(allocator_api)]

use std::alloc::{Alloc, AllocError, Allocator, Layout, System};
use std::ptr::NonNull;

use std::boxed::Box;

struct LeakAlloc {}

unsafe impl Alloc for LeakAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        System.alloc_ref().allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        System.alloc_ref().deallocate(ptr, layout)
    }
}

unsafe impl Allocator for LeakAlloc {
    type Alloc = Self;
    fn alloc_ref(&self) -> &Self::Alloc {
        self
    }
}

fn use_value(_: u32) {}

fn main() {
    let alloc = LeakAlloc {};
    let boxed = Box::new_in(10, alloc.by_ref());
    let theref = Box::leak(boxed);
    drop(alloc);
    //~^ ERROR cannot move out of `alloc` because it is borrowed
    use_value(*theref)
}
