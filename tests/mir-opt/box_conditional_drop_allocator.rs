// skip-filecheck
//@ test-mir-pass: ElaborateDrops
//@ needs-unwind
#![feature(allocator_api)]

// Regression test for #131082.
// Testing that the allocator of a Box is dropped in conditional drops

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;

struct DropAllocator;

unsafe impl Allocator for DropAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout);
    }
}
impl Drop for DropAllocator {
    fn drop(&mut self) {}
}

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

// EMIT_MIR box_conditional_drop_allocator.main.ElaborateDrops.diff
fn main() {
    let b = Box::new_in(HasDrop, DropAllocator);
    if true {
        drop(*b);
    } else {
        drop(b);
    }
}
