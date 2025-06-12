//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// FIXME(fmease): I'd prefer to report a cycle error here instead of an overflow one.

struct T;

impl T {
    type This = Self::This;
    //[current]~^ ERROR overflow evaluating associated type `T::This`
    //[next]~^^ ERROR type mismatch resolving `T::This normalizes-to _`
}

fn main() {}
