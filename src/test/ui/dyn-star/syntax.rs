// Make sure we can parse the `dyn* Trait` syntax
//
// check-pass

#![feature(dyn_star)]
#![allow(incomplete_features)]

pub fn dyn_star_parameter(_: dyn* Send) {
}

fn main() {}
