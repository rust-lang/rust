// compile-flags: -Ztrait-solver=next
#![feature(rustc_attrs, trivial_bounds)]

// We have to be careful here:
//
// We either have the provisional result of `A -> B -> A` on the
// stack, which is a fully coinductive cycle. Accessing the
// provisional result for `B` as part of the `A -> C -> B -> A` cycle
// has to make sure we don't just use the result of `A -> B -> A` as the
// new cycle is inductive.
//
// Alternatively, if we have `A -> C -> A` first, then `A -> B -> A` has
// a purely inductive stack, so something could also go wrong here.

#[rustc_coinductive]
trait A {}
#[rustc_coinductive]
trait B {}
trait C {}

impl<T: B + C> A for T {}
impl<T: A> B for T {}
impl<T: B> C for T {}

fn impls_a<T: A>() {}

// The same test with reordered where clauses to make sure we're actually testing anything.
#[rustc_coinductive]
trait AR {}
#[rustc_coinductive]
trait BR {}
trait CR {}

impl<T: CR + BR> AR for T {}
impl<T: AR> BR for T {}
impl<T: BR> CR for T {}

fn impls_ar<T: AR>() {}

fn main() {
    impls_a::<()>();
    //~^ ERROR overflow evaluating the requirement `(): A`

    impls_ar::<()>();
    //~^ ERROR overflow evaluating the requirement `(): AR`
}
