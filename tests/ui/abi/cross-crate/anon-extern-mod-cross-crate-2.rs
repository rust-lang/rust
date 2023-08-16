//@run
//@aux-build:anon-extern-mod-cross-crate-1.rs
// pretty-expanded FIXME #23616
//@ignore-target-wasm32-unknown-unknown no libc to test ffi with

extern crate anonexternmod;

use anonexternmod::rust_get_test_int;

pub fn main() {
    unsafe {
        rust_get_test_int();
    }
}
