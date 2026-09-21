#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]

const INC<const N: usize>: usize = core::direct_const_arg!(const { N + 1 });
//~^ ERROR generic parameters may not be used in const operations
//~| HELP add `#![feature(generic_const_args)]`

fn main() {}
