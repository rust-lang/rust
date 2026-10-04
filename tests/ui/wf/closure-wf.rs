//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

trait Bound {}
struct NeedsBound<T: Bound>(T);

// Checks that we enforce that closure args are WF.

fn constrain_inner<T, F: for<'a> FnOnce(&'a (), NeedsBound<T>)>(_: T, _: F) {}
//~^ WARN the trait bound `T: Bound` is not satisfied
//~| WARN this was previously accepted by the compiler
//[next]~| WARN the trait bound `T: Bound` is not satisfied
//[next]~| WARN this was previously accepted by the compiler

fn main() {
    constrain_inner(1u32, |&(), _| ());
    //~^ ERROR the trait bound `u32: Bound` is not satisfied
}
