// run-pass
#![allow(improper_ctypes)]

// ignore-wasm32-bare no libc to test ffi with

pub struct TwoU64s {
    one: u64,
    two: u64,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_return_TwoU64s() -> TwoU64s;
}

pub fn main() {
    unsafe {
        let y = rust_dbg_extern_return_TwoU64s();
        assert_eq!(y.one, 10);
        assert_eq!(y.two, 20);
    }
}
