#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]

use std::gca;

const INC<const N: usize>: usize = gca!(const { N + 1 });
//~^ ERROR generic parameters may not be used in const operations
//~| HELP add `#![feature(generic_const_args)]`

fn main() {}
