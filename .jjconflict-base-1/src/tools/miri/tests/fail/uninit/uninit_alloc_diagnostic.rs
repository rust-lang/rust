//@compile-flags: -Zmiri-disable-validation
//@error-in-other-file: memory is uninitialized at [0x4..0x10]

#![allow(dropping_copy_types)]

use std::alloc::{Layout, alloc, dealloc};
use std::slice::from_raw_parts;

fn main() {
    let layout = Layout::from_size_align(32, 8).unwrap();
    unsafe {
        let ptr = alloc(layout);
        *ptr = 0x41;
        *ptr.add(1) = 0x42;
        *ptr.add(2) = 0x43;
        *ptr.add(3) = 0x44;
        *ptr.add(16) = 0x00;
        let slice1 = from_raw_parts(ptr, 16);
        let slice2 = from_raw_parts(ptr.add(16), 16);
        drop(slice1.cmp(slice2));
        dealloc(ptr, layout);
    }
}
