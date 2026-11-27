//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::read_volatile requires
//@ revisions: misaligned

use std::ptr;

fn main() {
    let src = [0u16; 2];
    let src = src.as_ptr();
    unsafe {
        #[cfg(misaligned)]
        ptr::read_volatile(src.byte_add(1));
    }
}
