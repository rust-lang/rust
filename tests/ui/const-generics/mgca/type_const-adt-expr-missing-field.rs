// Regression test for https://github.com/rust-lang/rust/issues/154632

#![feature(generic_const_exprs)]
#![feature(gca_min_const_items)]
#![feature(gca_macroless_args)]
#![feature(generic_const_items)]

use std::gca;

const ADD1<const N: usize>: usize = gca!(const { N + 1 });
//~^ ERROR: unconstrained generic constant
const AliasFnUnused: ADD1 = gca!(ADD1::<{ Some::<usize> {} }>);
//~^ ERROR: cannot find type `ADD1` in this scope [E0573]
//~| ERROR: struct expression with missing field initialiser for `0`

fn main() {}
