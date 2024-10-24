//@ run-pass
#![feature(allocator_api)]

// Regression test for #131082.
// Testing that the allocator of a Box is dropped in conditional drops

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cell::Cell;
use std::ptr::NonNull;

struct DropCheckingAllocator<'a>(&'a Cell<bool>);

unsafe impl Allocator for DropCheckingAllocator<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout);
    }
}
impl Drop for DropCheckingAllocator<'_> {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {
    let dropped = Cell::new(false);
    {
        let b = Box::new_in(HasDrop, DropCheckingAllocator(&dropped));
        if true {
            drop(*b);
        } else {
            drop(b);
        }
    }
    assert!(dropped.get());
}
