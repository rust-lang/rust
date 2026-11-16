//! Regression test for https://github.com/rust-lang/rust/issues/163310.
//! `unused_qualifications` must not suggest removing a qualification when the
//! unqualified path would be ambiguous because of multiple glob imports.

//@ check-pass

#![deny(unused_qualifications)]
#![allow(dead_code)]

pub type T = String;

mod mod_a {
    pub type T = bool;
    pub const C: i32 = 42;
}

mod mod_b {
    use crate::mod_a::*;
    use crate::*;

    type T2 = mod_a::T;
    const C2: i32 = C;
}

fn main() {}
