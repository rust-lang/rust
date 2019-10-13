// run-pass
// ABI is cdecl by default

// ignore-wasm32-bare no libc to test ffi with
// pretty-expanded FIXME #23616

#![feature(rustc_private)]

mod rustrt {
    extern crate libc;

    #[link(name = "rust_test_helpers", kind = "static")]
    extern {
        pub fn rust_get_test_int() -> libc::intptr_t;
    }
}

pub fn main() {
    unsafe {
        rustrt::rust_get_test_int();
    }
}
