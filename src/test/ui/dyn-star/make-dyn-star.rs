// run-pass
#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn make_dyn_star(i: usize) {
    let _dyn_i: dyn* Debug = i;
}

fn make_dyn_star_explicit(i: usize) {
    let _dyn_i: dyn* Debug = i as dyn* Debug;
}

fn main() {
    make_dyn_star(42);
    make_dyn_star_explicit(42);
}
