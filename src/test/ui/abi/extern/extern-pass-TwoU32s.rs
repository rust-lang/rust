// run-pass
#![allow(improper_ctypes)]

// ignore-wasm32-bare no libc for ffi testing

// Test a foreign function that accepts and returns a struct
// by value.

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TwoU32s {
    one: u32,
    two: u32,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_TwoU32s(v: TwoU32s) -> TwoU32s;
}

pub fn main() {
    unsafe {
        let x = TwoU32s { one: 22, two: 23 };
        let y = rust_dbg_extern_identity_TwoU32s(x);
        assert_eq!(x, y);
    }
}
