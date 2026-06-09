//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

#![feature(inherent_associated_types, rustc_attrs)]
#![allow(incomplete_features)]
#![rustc_no_implicit_bounds]

// FIXME(fmease): I'd prefer to report a cycle error here instead of an overflow one.

struct T;

impl T {
    type This = Self::This;
    //[current]~^ ERROR: overflow evaluating associated type `T::This`
    //[next]~^^ ERROR: type mismatch resolving `T::This normalizes-to _`
}

fn main() {}
