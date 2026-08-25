//! Regression test for https://github.com/rust-lang/rust/issues/3668
//!
//@ run-rustfix
#![allow(unused_variables, dead_code)]
fn f(x: isize) {
    static child: isize = x + 1;
    //~^ ERROR attempt to use a non-constant value in a constant
}

fn main() {}
