//@ run-pass

// Static recursion check shouldn't fail when given a foreign item (#18279)

//@ aux-build:check_static_recursion_foreign_helper.rs

//@ pretty-expanded FIXME #23616

#![feature(rustc_private)]

extern crate check_static_recursion_foreign_helper;
extern crate libc;

use libc::c_int;

extern "C" {
    static test_static: c_int;
}

pub static B: &'static c_int = unsafe { &test_static };

pub fn main() {}
