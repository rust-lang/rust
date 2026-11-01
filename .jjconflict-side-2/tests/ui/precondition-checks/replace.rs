//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::replace requires
//@ revisions: null misaligned

#![allow(invalid_null_arguments)]

use std::ptr;

fn main() {
    let mut dst = [0u16; 2];
    let dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(null)]
        ptr::replace(ptr::null_mut::<u8>(), 1);
        #[cfg(misaligned)]
        ptr::replace(dst.byte_add(1), 1u16);
    }
}
