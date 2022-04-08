// Make sure we can parse the `dyn* Trait` syntax
//
// check-pass


#![feature(dyn_star)]

pub fn dyn_star_parameter(_: &dyn* Send) {
}

fn main() {}
