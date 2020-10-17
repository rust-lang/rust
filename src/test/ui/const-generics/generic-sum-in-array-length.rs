// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

fn foo<const A: usize, const B: usize>(bar: [usize; A + B]) {}
//[min]~^ ERROR generic parameters may not be used in const operations
//[min]~| ERROR generic parameters may not be used in const operations
//[full]~^^^ ERROR constant expression depends on a generic parameter

fn main() {}
