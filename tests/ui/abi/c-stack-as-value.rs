//@ run-pass
//@ pretty-expanded FIXME #23616

#![feature(rustc_private)]

mod rustrt {
    extern crate libc;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub fn main() {
    let _foo = rustrt::rust_get_test_int;
}
