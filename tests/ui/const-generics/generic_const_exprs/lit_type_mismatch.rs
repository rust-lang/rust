//! ICE regression test for #114317 and #126182
//! Type mismatches of literals cause errors int typeck,
//! but those errors cannot be propagated to the various
//! `lit_to_const` call sites. Now `lit_to_const` just delays
//! a bug and produces an error constant on its own.

#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct A<const B: () = 1, C>(C);
//~^ ERROR: generic parameters with a default must be trailing
//~| ERROR: mismatched types

struct Cond<const B: bool>;

struct Thing<T = Cond<0>>(T);
//~^ ERROR: mismatched types

impl Thing {}

fn main() {}
