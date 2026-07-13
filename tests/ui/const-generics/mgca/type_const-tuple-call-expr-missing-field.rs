#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![feature(adt_const_params)]
#![feature(generic_const_items)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Wrap(usize);

type const ADD1<const N: usize>: usize = const { N + 1 };
//~^ ERROR: unconstrained generic constant
type const AliasFnUnused: ADD1 = ADD1::<{ Wrap(Some::<usize> {}) }>;
//~^ ERROR: expected type, found constant `ADD1` [E0573]
//~| ERROR: struct expression with missing field initialiser for `0`

fn main() {}
