//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::copy_nonoverlapping requires
//@ revisions: null_src null_dst misaligned_src misaligned_dst overlapping oversized

#![allow(invalid_null_arguments)]

use std::ptr;

fn main() {
    let src = [0u16; 3];
    let mut dst = [0u16; 3];
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(null_src)]
        ptr::copy_nonoverlapping(ptr::null(), dst, 1);
        #[cfg(null_dst)]
        ptr::copy_nonoverlapping(src, ptr::null_mut(), 1);
        #[cfg(misaligned_src)]
        ptr::copy_nonoverlapping(src.byte_add(1), dst, 1);
        #[cfg(misaligned_dst)]
        ptr::copy_nonoverlapping(src, dst.byte_add(1), 1);
        #[cfg(overlapping)]
        ptr::copy_nonoverlapping(dst, dst.add(1), 2);
        #[cfg(oversized)]
        {
            let count = isize::MAX as usize + 1;
            let src = ptr::without_provenance::<u8>(1);
            let dst = ptr::without_provenance_mut::<u8>(1 + count);
            ptr::copy_nonoverlapping(src, dst, count);
        }
    }
}
