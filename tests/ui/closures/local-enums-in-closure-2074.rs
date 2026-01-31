//! Regression test for https://github.com/rust-lang/rust/issues/2074

//@ run-pass

#![allow(non_camel_case_types)]

pub fn main() {
    let one = || {
        enum r {
            a,
        }
        r::a as usize
    };
    let two = || {
        enum r {
            a,
        }
        r::a as usize
    };
    one();
    two();
}
