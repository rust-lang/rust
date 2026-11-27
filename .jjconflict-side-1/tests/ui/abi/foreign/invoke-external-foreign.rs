//@ run-pass
//@ aux-build:foreign_lib.rs

// The purpose of this test is to check that we can
// successfully (and safely) invoke external, cdecl
// functions from outside the crate.


extern crate foreign_lib;

pub fn main() {
    unsafe {
        let _foo = foreign_lib::rustrt::rust_get_test_int();
    }
}
