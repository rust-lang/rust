//@ run-pass
#![allow(improper_ctypes)]

pub struct TwoU8s {
    one: u8,
    two: u8,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_return_TwoU8s() -> TwoU8s;
}

pub fn main() {
    unsafe {
        let y = rust_dbg_extern_return_TwoU8s();
        assert_eq!(y.one, 10);
        assert_eq!(y.two, 20);
    }
}
