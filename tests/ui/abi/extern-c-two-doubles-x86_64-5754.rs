// https://github.com/rust-lang/rust/issues/5754
//@ build-pass
#![allow(dead_code)]
#![allow(improper_ctypes)]

struct TwoDoubles {
    r: f64,
    i: f64
}

extern "C" {
    fn rust_dbg_extern_identity_TwoDoubles(arg1: TwoDoubles) -> TwoDoubles;
}

pub fn main() {}
