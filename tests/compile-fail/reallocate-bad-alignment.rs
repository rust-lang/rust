#![feature(alloc, heap_api)]

extern crate alloc;

// error-pattern: tried to access memory with alignment 1, but alignment 2 is required

use alloc::heap::*;
fn main() {
    unsafe {
        let x = allocate(1, 1);
        let _y = reallocate(x, 1, 1, 2);
    }
}
