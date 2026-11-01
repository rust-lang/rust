//@ run-pass

// Static recursion check shouldn't fail when given a foreign item (#18279)

//@ aux-build:check_static_recursion_foreign_helper.rs


extern crate check_static_recursion_foreign_helper;

use std::ffi::c_int;

extern "C" {
    static test_static: c_int;
}

pub static B: &'static c_int = unsafe { &test_static };

pub fn main() {}
