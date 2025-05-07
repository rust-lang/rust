//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::read_volatile requires
//@ revisions: null misaligned

#![allow(invalid_null_arguments)]

use std::ptr;

fn main() {
    let src = [0u16; 2];
    let src = src.as_ptr();
    unsafe {
        #[cfg(null)]
        ptr::read_volatile(ptr::null::<u8>());
        #[cfg(misaligned)]
        ptr::read_volatile(src.byte_add(1));
    }
}
