//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]

// A test intended to check how we handle provisional results
// for a goal computed with an inductive and a coinductive stack.
//
// Unfortunately this doesn't really detect whether we've done
// something wrong but instead only showcases that we thought of
// this.
//
// FIXME(-Znext-solver=coinductive): With the new coinduction approach
// the same goal stack can be both inductive and coinductive, depending
// on why we're proving a specific nested goal. Rewrite this test
// at that point instead of relying on `BInd`.


#[rustc_coinductive]
trait A {}

#[rustc_coinductive]
trait B {}
trait BInd {}
impl<T: ?Sized + B> BInd for T {}

#[rustc_coinductive]
trait C {}
trait CInd {}
impl<T: ?Sized + C> CInd for T {}

impl<T: ?Sized + BInd + C> A for T {}
impl<T: ?Sized + CInd + C> B for T {}
impl<T: ?Sized + B + A> C for T {}

fn impls_a<T: A>() {}

fn main() {
    impls_a::<()>();
    //~^ ERROR overflow evaluating the requirement `(): A`
}
