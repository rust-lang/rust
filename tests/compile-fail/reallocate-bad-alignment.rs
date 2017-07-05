#![feature(alloc, heap_api)]

extern crate alloc;

// error-pattern: tried to deallocate or reallocate using incorrect alignment or size

use alloc::heap::*;
fn main() {
    unsafe {
        let x = allocate(1, 1);
        let _y = reallocate(x, 1, 1, 2);
    }
}
