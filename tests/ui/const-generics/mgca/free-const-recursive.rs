//@ revisions: min_gca gca
//@ check-fail
//@ compile-flags: -Znext-solver=globally

// Regression test for <https://github.com/rust-lang/rust/issues/153831>

#![feature(gca_min_const_items)]
#![expect(incomplete_features)]
#![cfg_attr(gca, feature(gca_const_items))]

use std::gca;

const A: () = gca!(A);
//[gca]~^ ERROR: overflow evaluating the requirement `A == _`
//[gca]~| ERROR: overflow evaluating the requirement `the constant `A` has type `()``
//[min_gca]~^^^ ERROR: cycle detected when computing the type-level value for `A`

fn main() {
    A;
}
