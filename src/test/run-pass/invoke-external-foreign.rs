// aux-build:foreign_lib.rs
// ignore-wasm32-bare no libc to test ffi with

// The purpose of this test is to check that we can
// successfully (and safely) invoke external, cdecl
// functions from outside the crate.

// pretty-expanded FIXME #23616

extern crate foreign_lib;

pub fn main() {
    unsafe {
        let _foo = foreign_lib::rustrt::rust_get_test_int();
    }
}
