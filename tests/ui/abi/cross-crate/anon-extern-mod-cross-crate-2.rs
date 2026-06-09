//@ run-pass
//@ aux-build:anon-extern-mod-cross-crate-1.rs

extern crate anonexternmod;

use anonexternmod::rust_get_test_int;

pub fn main() {
    unsafe {
        rust_get_test_int();
    }
}
