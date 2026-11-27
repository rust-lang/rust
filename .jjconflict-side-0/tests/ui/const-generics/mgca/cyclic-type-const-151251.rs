//@ needs-rustc-debug-assertions

#![feature(gca_min_const_items)]
#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

use std::gca;

const A: u8 = gca!(A);
//~^ ERROR cycle detected when computing the type-level value for `A`

fn main() {}
