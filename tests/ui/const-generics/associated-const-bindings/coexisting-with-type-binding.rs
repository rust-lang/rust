// Regression test for issue #112560.
// Respect the fact that (associated) types and constants live in different namespaces and
// therefore equality bounds involving identically named associated items don't conflict if
// their kind (type vs. const) differs. This obviously extends to supertraits.

//@ check-pass

#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![allow(incomplete_features)]

trait Trait: SuperTrait {
    type N;
    type Q;

    #[rustc_always_gca]
    const N: usize;
}

trait SuperTrait {
    #[rustc_always_gca]
    const Q: &'static str;
}

fn take0(_: impl Trait<N = 0, N = ()>) {}

fn take1(_: impl Trait<Q = { core::direct_const_arg!("...") }, Q = [()]>) {}

fn main() {}
