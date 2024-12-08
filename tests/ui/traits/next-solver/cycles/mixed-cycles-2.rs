//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]

// A test showcasing that the solver may need to
// compute a goal which is already in the provisional
// cache.
//
// However, given that `(): BInd` and `(): B` are currently distinct
// goals, this is actually not possible right now.
//
// FIXME(-Znext-solver=coinductive): With the new coinduction approach
// the same goal stack can be both inductive and coinductive, depending
// on why we're proving a specific nested goal. Rewrite this test
// at that point.

#[rustc_coinductive]
trait A {}

#[rustc_coinductive]
trait B {}
trait BInd {}
impl<T: ?Sized + B> BInd for T {}

impl<T: ?Sized + BInd + B> A for T {}
impl<T: ?Sized + BInd> B for T {}

fn impls_a<T: A>() {}

fn main() {
    impls_a::<()>();
    //~^ ERROR overflow evaluating the requirement `(): A`
}
