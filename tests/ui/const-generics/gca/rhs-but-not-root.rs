//@ compile-flags: -Znext-solver
#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

// Anon consts must be the root of the RHS to be GCA.
const FOO<const N: usize>: usize = core::direct_const_arg!(ID::<const { N + 1 }>);
//~^ ERROR generic parameters in const blocks are not allowed; use a named `const` item instead
const ID<const N: usize>: usize = core::direct_const_arg!(N);

fn main() {}
