#![feature(alloc, heap_api)]

extern crate alloc;

// error-pattern: tried to deallocate with a pointer not to the beginning of an existing object

use alloc::heap::*;
fn main() {
    unsafe {
        let x = allocate(1, 1);
        deallocate(x, 1, 1);
        deallocate(x, 1, 1);
    }
}
