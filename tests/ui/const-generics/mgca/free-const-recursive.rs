//@ revisions: min_gca gca
//@ check-fail
//@ compile-flags: -Znext-solver=globally

// Regression test for <https://github.com/rust-lang/rust/issues/153831>

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]
#![cfg_attr(gca, feature(generic_const_args))]

const A: () = core::direct_const_arg!(A);
//[gca]~^ ERROR: overflow evaluating the requirement `A == _`
//[gca]~| ERROR: overflow evaluating the requirement `the constant `A` has type `()``
//[min_gca]~^^^ ERROR: cycle detected when computing the type-level value for `A`

fn main() {
    A;
}
