// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(improper_ctypes)]

// pretty-expanded FIXME #23616

struct TwoDoubles {
    r: f64,
    i: f64
}

extern "C" {
    fn rust_dbg_extern_identity_TwoDoubles(arg1: TwoDoubles) -> TwoDoubles;
}

pub fn main() {}
