#![feature(allocator_api)]

use std::{
    alloc::{AllocError, Allocator, Layout},
    any::Any,
    pin::Pin,
    ptr::NonNull,
};

struct BadAlloc;
unsafe impl Allocator for BadAlloc {
    fn allocate(&self, _: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unimplemented!()
    }
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {
        unimplemented!()
    }
}

fn main() {
    // no requirements for the allocator, we are just pinning a `Box<impl Unpin>`
    let base: Pin<Box<i32, BadAlloc>> = Pin::new(Box::new_in(1i32, BadAlloc));

    // unsize coercion (must fail)
    let _: Pin<Box<dyn Any, BadAlloc>> = base;
    //~^ ERROR: the trait bound `BadAlloc: StaticAllocator` is not satisfied [E0277]
}
