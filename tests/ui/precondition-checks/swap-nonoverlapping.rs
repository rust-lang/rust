//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::swap_nonoverlapping requires
//@ revisions: null_src null_dst misaligned_src misaligned_dst overlapping

#![allow(invalid_null_arguments)]

use std::ptr;

fn main() {
    let mut src = [0u16; 3];
    let mut dst = [0u16; 3];
    let src = src.as_mut_ptr();
    let dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(null_src)]
        ptr::swap_nonoverlapping(ptr::null_mut(), dst, 1);
        #[cfg(null_dst)]
        ptr::swap_nonoverlapping(src, ptr::null_mut(), 1);
        #[cfg(misaligned_src)]
        ptr::swap_nonoverlapping(src.byte_add(1), dst, 1);
        #[cfg(misaligned_dst)]
        ptr::swap_nonoverlapping(src, dst.byte_add(1), 1);
        #[cfg(overlapping)]
        ptr::swap_nonoverlapping(dst, dst.add(1), 2);
    }
}
