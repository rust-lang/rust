//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::write_volatile requires
//@ revisions: misaligned

use std::ptr;

fn main() {
    let mut dst = [0u16; 2];
    let mut dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(misaligned)]
        ptr::write_volatile(dst.byte_add(1), 1u16);
    }
}
