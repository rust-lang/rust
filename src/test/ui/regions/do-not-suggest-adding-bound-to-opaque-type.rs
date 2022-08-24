#![feature(allocator_api)]

use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::NonNull,
};

struct GhostBump;

unsafe impl Allocator for &GhostBump {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        todo!()
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        todo!()
    }
}

fn foo() -> impl Iterator<Item = usize> {
    let arena = GhostBump;
    let mut vec = Vec::new_in(&arena); //~ ERROR `arena` does not live long enough
    vec.push(1);
    vec.push(2);
    vec.push(3);
    vec.into_iter()
}

fn main() {}
