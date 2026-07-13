// Regression test for https://github.com/rust-lang/rust/issues/154632

#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![feature(generic_const_items)]

type const ADD1<const N : usize> : usize = const { N + 1 };
//~^ ERROR: unconstrained generic constant
type const AliasFnUnused: ADD1 = ADD1::<{ Some::<usize> {} }>;
//~^ ERROR: expected type, found constant `ADD1` [E0573]
//~| ERROR: struct expression with missing field initialiser for `0`

fn main() {}
