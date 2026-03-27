//! Regression test for <https://github.com/rust-lang/rust/issues/152653>
//@ incremental
#![feature(min_generic_const_args)]
type const FOO: usize = 1_i32; //~ ERROR: the constant `1` is not of type `usize`
type const BAR: usize = -1_i32; //~ ERROR: the constant `-1` is not of type `usize`

fn main() {
    FOO;
    BAR;
}
