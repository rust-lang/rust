//@ compile-flags: --edition 2021
//@ failure-status: 101
use core::ffi;

extern "C" {
    fn printf(ptr: *const ffi::c_char, ...) -> ffi::c_int;
}

fn generic_printf<T>(c: &ffi::CStr, arg: T) {
    unsafe { printf(c.as_ptr(), arg) };
}

fn main() {
    generic_printf(c"%d", 2u8);
    generic_printf(c"%f", 3.333_f32);
    generic_printf(c"%s", vec![6, 2, 8, 3, 1, 8, 5, 3, 0]);
}
