//@ compile-flags: -Znext-solver

// Proving `W<?0>: Trait` instantiates `?0` with `(W<?1>, W<?2>)` and then
// proves `W<?1>: Trait` and `W<?2>: Trait`, resulting in a coinductive cycle.
//
// Proving coinductive cycles runs until we reach a fixpoint. However, after
// computing `try_evaluate_added_goals` in the second fixpoint iteration, the
// self type already has a depth equal to the number of steps. This results
// in enormous constraints, causing the canonicalizer to hang without ever
// reaching the recursion limit. We currently avoid that by erasing the constraints
// from overflow.
//
// This previously caused a hang in the trait solver, see
// https://github.com/rust-lang/trait-system-refactor-initiative/issues/13.

#![feature(rustc_attrs)]

#[rustc_coinductive]
trait Trait {}

struct W<T>(T);

impl<T, U> Trait for W<(W<T>, W<U>)>
where
    W<T>: Trait,
    W<U>: Trait,
{
}

fn impls<T: Trait>() {}

fn main() {
    impls::<W<_>>();
    //~^ ERROR overflow evaluating the requirement
}
