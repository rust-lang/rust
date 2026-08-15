//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: ptr::write requires
//@ revisions: null misaligned
//@ ignore-test (unimplemented)

use std::ptr;

fn main() {
    let mut dst = [0u16; 2];
    let mut dst = dst.as_mut_ptr();
    unsafe {
        #[cfg(null)]
        ptr::write_bytes(ptr::null_mut::<u8>(), 1u8, 2);
        #[cfg(misaligned)]
        ptr::write_bytes(dst.byte_add(1), 1u8, 2);
    }
}
