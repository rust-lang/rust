#![feature(c_variadic)]

use core::ffi::VaList;

fn helper(mut ap: VaList) -> i32 {
    unsafe { ap.arg::<i32>() }
}

unsafe extern "C" fn variadic(a: i32, ap: ...) -> i32 {
    assert_eq!(a, 42);
    helper(ap)
}

fn main() {
    assert_eq!(unsafe { variadic(42, 1) }, 1);
}
