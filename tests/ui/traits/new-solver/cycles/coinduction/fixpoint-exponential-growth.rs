// compile-flags: -Ztrait-solver=next

// Proving `W<?0>: Trait` instantiates `?0` with `(W<?1>, W<?2>)` and then
// proves `W<?1>: Trait` and `W<?2>: Trait`, resulting in a coinductive cycle.
//
// Proving coinductive cycles runs until we reach a fixpoint. This fixpoint is
// never reached here and each step doubles the amount of nested obligations.
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
