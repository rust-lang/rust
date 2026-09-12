//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Do not emit the UNSATISFIED_HRTB_ARG FCW if we have ill-formed types.

#![allow(dead_code)]

trait Bound {}
struct W<'a, T: Bound>(&'a T);

fn errors<T>() where W<'static, T>: Sized, for<'a> W<'a, T>: Sized {}
//~^ ERROR the trait bound `T: Bound` is not satisfied

fn main() {}
