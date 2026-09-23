#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

use std::gca;

const A: u8 = gca!(A);
//~^ ERROR: cycle detected when computing the type-level value for `A` [E0391]

fn main() {}
