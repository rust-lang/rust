//! Regression test for <https://github.com/rust-lang/rust/issues/152653>
//!                     <https://github.com/rust-lang/rust/issues/154636>
//@ incremental
#![feature(min_generic_const_args)]
const R: usize = core::direct_const_arg!(1_i32); //~ ERROR: the constant `1` is not of type `usize`
const U: usize = core::direct_const_arg!(-1_i32); //~ ERROR: the constant `-1` is not of type `usize`
const S: bool = core::direct_const_arg!(1i32); //~ ERROR: the constant `1` is not of type `bool`
const T: bool = core::direct_const_arg!(-1i32); //~ ERROR: the constant `-1` is not of type `bool`

fn main() {
    R;
    U;
    S;
    T;
}
