// Regression test for #119316
// This used to ICE with "expected type of closure to be a closure"
// when using a closure as a const generic default with generic_const_exprs.
// Now it correctly reports a type mismatch error.

#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

struct Bug<const N: usize = { || 0 }>;
//~^ ERROR mismatched types

fn main() {}
