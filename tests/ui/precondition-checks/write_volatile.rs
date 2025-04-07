//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::write_volatile requires
//@ revisions: null misaligned

#![allow(invalid_null_arguments)]

use std::ptr;

fn main() {
    let mut dst = [0u16; 2];
    let mut dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(null)]
        ptr::write_volatile(ptr::null_mut::<u8>(), 1u8);
        #[cfg(misaligned)]
        ptr::write_volatile(dst.byte_add(1), 1u16);
    }
}
