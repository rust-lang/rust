// check-pass
#![feature(dyn_star)]

use std::fmt::Debug;

pub fn dyn_star_parameter(_: dyn* Send) {
}

fn make_dyn_star() {
    let i = 42usize;
    let dyn_i: dyn* Debug = i as dyn* Debug;
}

fn main() {}
