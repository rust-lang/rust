#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

// Anon consts must be the root of the RHS to be GCA.
type const FOO<const N: usize>: usize = ID::<const { N + 1 }>;
//~^ ERROR generic parameters in const blocks are only allowed as the direct value of a `type const`
type const ID<const N: usize>: usize = N;

fn main() {}
