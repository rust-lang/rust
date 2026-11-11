//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::copy requires
//@ revisions: null_src null_dst misaligned_src misaligned_dst

#![allow(invalid_null_arguments)]

use std::ptr;

fn main() {
    let src = [0u16; 3];
    let mut dst = [0u16; 3];
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(null_src)]
        ptr::copy(ptr::null(), dst, 1);
        #[cfg(null_dst)]
        ptr::copy(src, ptr::null_mut(), 1);
        #[cfg(misaligned_src)]
        ptr::copy(src.byte_add(1), dst, 1);
        #[cfg(misaligned_dst)]
        ptr::copy(src, dst.byte_add(1), 1);
    }
}
