#![allow(unused)]
#![warn(clippy::fn_null_check)]
#![allow(clippy::cmp_null)]
#![allow(clippy::ptr_eq)]
#![allow(clippy::zero_ptr)]

pub const ZPTR: *const () = 0 as *const _;
pub const NOT_ZPTR: *const () = 1 as *const _;

fn main() {
    let fn_ptr = main;

    if (fn_ptr as *mut ()).is_null() {}
    if (fn_ptr as *const u8).is_null() {}
    if (fn_ptr as *const ()) == std::ptr::null() {}
    if (fn_ptr as *const ()) == (0 as *const ()) {}
    if (fn_ptr as *const ()) == ZPTR {}

    // no lint
    if (fn_ptr as *const ()) == NOT_ZPTR {}
}
