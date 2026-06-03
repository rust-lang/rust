//@ build-pass
//@ compile-flags: -Cdebuginfo=2
// fixes issue #94725

#![feature(allocator_api)]

use std::alloc::{Alloc, AllocError, Allocator, Layout};
use std::ptr::NonNull;

struct ZST;

unsafe impl Alloc for &ZST {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        todo!()
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        todo!()
    }
}

unsafe impl Allocator for &ZST {
    type Alloc = Self;
    fn alloc_ref(&self) -> &Self::Alloc {
        self
    }
}

fn main() {
    let _ = Box::<i32, &ZST>::new_in(43, &ZST);
}
