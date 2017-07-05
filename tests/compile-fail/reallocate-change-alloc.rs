#![feature(alloc, heap_api)]

extern crate alloc;

use alloc::heap::*;
fn main() {
    unsafe {
        let x = allocate(1, 1);
        let _y = reallocate(x, 1, 1, 1);
        let _z = *x; //~ ERROR: dangling pointer was dereferenced
    }
}
