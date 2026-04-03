//@ run-pass
#![allow(improper_ctypes)]

// Test a foreign function that accepts and returns a struct
// by value.

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TwoU8s {
    one: u8,
    two: u8,
}

#[cfg_attr(target_env = "pauthtest", link(name = "rust_test_helpers", kind = "dylib"))]
#[cfg_attr(not(target_env = "pauthtest"), link(name = "rust_test_helpers", kind = "static"))]
extern "C" {
    pub fn rust_dbg_extern_identity_TwoU8s(v: TwoU8s) -> TwoU8s;
}

pub fn main() {
    unsafe {
        let x = TwoU8s { one: 22, two: 23 };
        let y = rust_dbg_extern_identity_TwoU8s(x);
        assert_eq!(x, y);
    }
}
