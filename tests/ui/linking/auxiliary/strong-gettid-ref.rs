// This crate must sort after `std` to reproduce #154439.

//@ no-prefer-dynamic
//@ compile-flags: -Copt-level=3

#![crate_type = "rlib"]

unsafe extern "C" {
    safe fn gettid() -> i32;
}

pub fn tid() -> i32 {
    gettid()
}
