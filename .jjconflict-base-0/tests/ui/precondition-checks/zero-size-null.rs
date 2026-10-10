// Test that none of the precondition checks panic on zero-sized reads or writes through null.

//@ run-pass
//@ compile-flags: -Zmir-opt-level=0 -Copt-level=0 -Cdebug-assertions=yes

use std::ptr;

fn main() {
    unsafe {
        #[expect(invalid_null_arguments)] // false-positive, copy of 0
        ptr::copy_nonoverlapping::<u8>(ptr::null(), ptr::null_mut(), 0);
        ptr::copy_nonoverlapping::<()>(ptr::null(), ptr::null_mut(), 123);
        #[expect(invalid_null_arguments)] // false-positive, copy of 0
        ptr::copy::<u8>(ptr::null(), ptr::null_mut(), 0);
        ptr::copy::<()>(ptr::null(), ptr::null_mut(), 123);
        ptr::swap::<()>(ptr::null_mut(), ptr::null_mut());
        ptr::replace::<()>(ptr::null_mut(), ());
        ptr::read::<()>(ptr::null());
        ptr::write::<()>(ptr::null_mut(), ());
        ptr::read_volatile::<()>(ptr::null());
        ptr::write_volatile::<()>(ptr::null_mut(), ());
    }
}
