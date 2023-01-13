#![crate_name = "anonexternmod"]
#![feature(rustc_private)]

extern crate libc;

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_get_test_int() -> libc::intptr_t;
}
