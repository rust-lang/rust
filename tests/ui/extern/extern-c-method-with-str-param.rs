//! Regression test for <https://github.com/rust-lang/rust/issues/28600>.
//! pub extern fn with parameter type &str inside struct impl caused ICE.
//@ build-pass

struct Test;

impl Test {
    #[allow(dead_code)]
    #[allow(unused_variables)]
    #[allow(improper_ctypes_definitions)]
    pub extern "C" fn test(val: &str) {

    }
}

fn main() {}
