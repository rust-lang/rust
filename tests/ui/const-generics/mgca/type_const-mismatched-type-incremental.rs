//! Regression test for <https://github.com/rust-lang/rust/issues/152653>
//!                     <https://github.com/rust-lang/rust/issues/154636>
//@ incremental
#![feature(min_generic_const_args)]
type const R: usize = 1_i32; //~ ERROR: the constant `1` is not of type `usize`
type const U: usize = -1_i32; //~ ERROR: the constant `-1` is not of type `usize`
type const S: bool = 1i32; //~ ERROR: the constant `1` is not of type `bool`
type const T: bool = -1i32; //~ ERROR: the constant `-1` is not of type `bool`

fn main() {
    R;
    U;
    S;
    T;
}
