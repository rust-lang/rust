#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

const A: u8 = gca!(A);
//~^ ERROR: cycle detected when computing the type-level value for `A` [E0391]

fn main() {}
