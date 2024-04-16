//@error-in-other-file: aborted
//@normalize-stderr-test: "unsafe \{ libc::abort\(\) \}|crate::intrinsics::abort\(\);" -> "ABORT();"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
#![feature(allocator_api)]

use std::alloc::*;
use std::ptr::NonNull;

struct BadAlloc;

// Create a failing allocator; Miri's native allocator never fails so this is the only way to
// actually call the alloc error handler.
unsafe impl Allocator for BadAlloc {
    fn allocate(&self, _l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        unreachable!();
    }
}

fn main() {
    let _b = Box::new_in(0, BadAlloc);
}
