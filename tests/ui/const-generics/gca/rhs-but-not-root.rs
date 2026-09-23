//@ compile-flags: -Znext-solver
#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

use std::gca;

// Anon consts must be the root of the RHS to be GCA.
const FOO<const N: usize>: usize = gca!(ID::<const { N + 1 }>);
//~^ ERROR generic parameters in const blocks are not allowed; use a named `const` item instead
const ID<const N: usize>: usize = gca!(N);

fn main() {}
