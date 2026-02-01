#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(opaque_generic_const_args)]
#![expect(incomplete_features)]

// Anon consts must be the root of the RHS to be OGCA.
#[type_const]
const FOO<const N: usize>: usize = ID::<const { N + 1 }>;
//~^ ERROR generic parameters may not be used in const operations

#[type_const]
const ID<const N: usize>: usize = N;

fn main() {}
