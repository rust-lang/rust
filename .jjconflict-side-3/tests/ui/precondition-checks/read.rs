//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::read requires
//@ revisions: null misaligned
//@ ignore-test (unimplemented)

use std::ptr;

fn main() {
    let src = [0u16; 2];
    let src = src.as_ptr();
    unsafe {
        #[cfg(null)]
        ptr::read(ptr::null::<u8>());
        #[cfg(misaligned)]
        ptr::read(src.byte_add(1));
    }
}
