// run-pass
#![allow(improper_ctypes)]

// ignore-wasm32-bare no libc for ffi testing

// Test a foreign function that accepts and returns a struct
// by value.

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TwoU64s {
    one: u64, two: u64
}

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    pub fn rust_dbg_extern_identity_TwoU64s(v: TwoU64s) -> TwoU64s;
}

pub fn main() {
    unsafe {
        let x = TwoU64s {one: 22, two: 23};
        let y = rust_dbg_extern_identity_TwoU64s(x);
        assert_eq!(x, y);
    }
}
