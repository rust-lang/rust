//@ run-pass
#![allow(improper_ctypes)]

// Test a foreign function that accepts and returns a struct by value.

// FiveU16s in particular is interesting because it is larger than a single 64 bit or 32 bit
// register, which are used as cast destinations on some targets, but does not evenly divide those
// sizes, causing there to be padding in the last element.

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct FiveU16s {
    one: u16,
    two: u16,
    three: u16,
    four: u16,
    five: u16,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn rust_dbg_extern_identity_FiveU16s(v: FiveU16s) -> FiveU16s;
}

pub fn main() {
    unsafe {
        let x = FiveU16s { one: 22, two: 23, three: 24, four: 25, five: 26 };
        let y = rust_dbg_extern_identity_FiveU16s(x);
        assert_eq!(x, y);
    }
}
