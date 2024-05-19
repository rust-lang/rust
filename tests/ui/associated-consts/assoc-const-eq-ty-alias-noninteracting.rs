// Regression test for issue #112560.
// Respect the fact that (associated) types and constants live in different namespaces and
// therefore equality bounds involving identically named associated items don't conflict if
// their kind (type vs. const) differs. This obviously extends to supertraits.

//@ check-pass

#![feature(associated_const_equality)]

trait Trait: SuperTrait {
    type N;
    type Q;

    const N: usize;
}

trait SuperTrait {
    const Q: &'static str;
}

fn take0(_: impl Trait<N = 0, N = ()>) {}

fn take1(_: impl Trait<Q = "...", Q = [()]>) {}

fn main() {}
