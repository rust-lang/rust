#![feature(generic_const_items, gca_min_const_items)]
#![expect(incomplete_features)]

use std::gca;

const INC<const N: usize>: usize = gca!(const { N + 1 });
//~^ ERROR generic parameters may not be used in const operations
//~| HELP add `#![feature(gca_const_items)]`

fn main() {}
