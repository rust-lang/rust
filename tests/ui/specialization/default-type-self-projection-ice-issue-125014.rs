//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[current] known-bug: #125014
//@[current] failure-status: 101
//@[current] dont-check-compiler-stderr

// Tests that we don't ICE when a `default type` is potentially used as a self-type in an impl.
// Regression for #125014.

#![feature(specialization)]
#![allow(incomplete_features)]

trait A {
    type B;
}

impl A for <u16 as A>::B {
    //[next]~^ ERROR the trait bound `u16: A` is not satisfied
    //[next]~^^ ERROR the trait bound `u16: A` is not satisfied
    //[next]~^^^ ERROR the trait bound `u16: A` is not satisfied
    //[next]~^^^^ ERROR the trait bound `u16: A` is not satisfied
    default type B = ();
    //[next]~^ ERROR the trait bound `u16: A` is not satisfied
}

fn main() {}
