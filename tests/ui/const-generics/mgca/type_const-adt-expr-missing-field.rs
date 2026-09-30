// Regression test for https://github.com/rust-lang/rust/issues/154632

#![feature(
    gca_adts,
    gca_macroless_args,
    gca_min_const_items,
    generic_const_exprs,
    generic_const_items,
    min_adt_const_params
)]

use std::gca;

const ADD1<const N: usize>: usize = gca!(const { N + 1 });
//~^ ERROR: unconstrained generic constant
const AliasFnUnused: ADD1 = gca!(ADD1::<{ Some::<usize> {} }>);
//~^ ERROR: cannot find type `ADD1` in this scope [E0573]
//~| ERROR: struct expression with missing field initialiser for `0`

fn main() {}
