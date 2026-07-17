#![feature(allocator_api)]

use std::{
    alloc::{AllocError, Allocator, Layout},
    pin::Pin,
    ptr::NonNull,
};

struct UntrustedAlloc;
unsafe impl Allocator for UntrustedAlloc {
    fn allocate(&self, _: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
}

pub fn main() {
    let _: Pin<Box<i32, UntrustedAlloc>> = Box::pin_in(1, UntrustedAlloc);
    //~^ ERROR: the trait bound `UntrustedAlloc: StaticAllocator` is not satisfied [E0277]
}
