//@ run-pass
#![allow(improper_ctypes)]

pub struct FiveU16s {
    one: u16,
    two: u16,
    three: u16,
    four: u16,
    five: u16,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_return_FiveU16s() -> FiveU16s;
}

pub fn main() {
    unsafe {
        let y = rust_dbg_extern_return_FiveU16s();
        assert_eq!(y.one, 10);
        assert_eq!(y.two, 20);
        assert_eq!(y.three, 30);
        assert_eq!(y.four, 40);
        assert_eq!(y.five, 50);
    }
}
