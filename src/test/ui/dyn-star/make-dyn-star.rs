// run-pass
#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn make_dyn_star(i: usize) {
    let _dyn_i: dyn* Debug = i;
}

fn main() {
    make_dyn_star(42);
}
