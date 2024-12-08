//@ run-pass
#![feature(dyn_star)]
#![allow(unused, incomplete_features)]

use std::fmt::Debug;

fn make_dyn_star() {
    let i = 42usize;
    let dyn_i: dyn* Debug = i;
}

fn main() {
    make_dyn_star();
}
