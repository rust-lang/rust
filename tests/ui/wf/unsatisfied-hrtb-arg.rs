//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Arguments of HRTBs must satisfy the trait bounds declared
// on the argument types. We currently emit a FCW for such
// code, as emitting a hard error would cause substantial
// crater breakage.
// See <https://github.com/rust-lang/rust/issues/162200>.

#![allow(dead_code)]

trait Bound {}
struct W<'a, T: Bound>(&'a T);

fn warns<T>() where for<'a> W<'a, T>: Sized {}
//~^ WARN the trait bound `T: Bound` is not satisfied
//~| WARN this was previously accepted by the compiler

// Adding the required bound silences the lint.
fn fixed<T: Bound>() where for<'a> W<'a, T>: Sized {}

#[allow(unsatisfied_hrtb_arg)]
fn allowed<T>() where for<'a> W<'a, T>: Sized {}

fn main() {}
