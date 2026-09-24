//! Regression test for <https://github.com/rust-lang/rust/issues/152653>
//!                     <https://github.com/rust-lang/rust/issues/154636>
//@ incremental
#![feature(min_generic_const_args)]

use std::gca;

const R: usize = gca!(1_i32); //~ ERROR: the constant `1` is not of type `usize`
const U: usize = gca!(-1_i32); //~ ERROR: the constant `-1` is not of type `usize`
const S: bool = gca!(1i32); //~ ERROR: the constant `1` is not of type `bool`
const T: bool = gca!(-1i32); //~ ERROR: the constant `-1` is not of type `bool`

fn main() {
    R;
    U;
    S;
    T;
}
