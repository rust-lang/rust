//@compile-flags: -Zoom=panic
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

struct Bomb;
impl Drop for Bomb {
    fn drop(&mut self) {
        eprintln!("yes we are unwinding!");
    }
}

fn main() {
    let bomb = Bomb;
    let _b = Box::new_in(0, BadAlloc);
    std::mem::forget(bomb); // defuse unwinding bomb
}
