//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//
// This tests asserts that CoerceUnsized must only appear in diagnostics if it's explicitly
// mentioned by the user code and not as a by product of next solver.
//
#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

trait Trait {}
fn require_coerce_unsized<T: CoerceUnsized<str>>(_: T) {}

fn main() {
    let _ = &1_i32 as &dyn Trait;
    //~^ ERROR the trait bound `i32: Trait` is not satisfied
    require_coerce_unsized(false);
    //~^ ERROR the trait bound `bool: CoerceUnsized<str>` is not satisfied
}
