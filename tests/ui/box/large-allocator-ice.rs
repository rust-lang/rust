//@ build-pass
#![feature(allocator_api)]
#![allow(unused_must_use)]

use std::alloc::{Alloc, Allocator};

struct BigAllocator([usize; 2]);

unsafe impl Alloc for BigAllocator {
    fn allocate(
        &self,
        _: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<u8>, std::alloc::AllocError> {
        todo!()
    }
    unsafe fn deallocate(&self, _: std::ptr::NonNull<u8>, _: std::alloc::Layout) {
        todo!()
    }
}

unsafe impl Allocator for BigAllocator {
    type Alloc = Self;
    fn alloc_ref(&self) -> &Self::Alloc {
        self
    }
}

fn main() {
    Box::new_in((), &std::alloc::Global);
    Box::new_in((), BigAllocator([0; 2]));
    generic_function(0);
}

fn generic_function<T>(val: T) {
    *Box::new_in(val, &std::alloc::Global);
}
