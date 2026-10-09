//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #149015

// A regression test for #149015. This ICEd in `fn rematch_impl` due to
// buggy cycle handling in the old solver.

trait Trait {
    type Assoc;
}

impl Trait for <&'static () as Trait>::Assoc {
//[next]~^ ERROR: the trait bound `&'static (): Trait` is not satisfied
//[next]~| ERROR: the trait bound `&'static (): Trait` is not satisfied
//[next]~| ERROR: the trait bound `&'static (): Trait` is not satisfied
//[next]~| ERROR: the trait bound `&'static (): Trait` is not satisfied
    type Assoc = ();
    //[next]~^ ERROR: the trait bound `&'static (): Trait` is not satisfied
}

fn main() {}
