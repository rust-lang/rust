#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]

#[type_const]
const INC<const N: usize>: usize = const { N + 1 };
//~^ ERROR generic parameters may not be used in const operations
//~| HELP add `#![feature(opaque_generic_const_args)]`

fn main() {}
