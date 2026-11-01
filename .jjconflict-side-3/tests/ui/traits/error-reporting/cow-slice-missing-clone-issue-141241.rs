//@ revisions: current next with_clone
//@ [next] compile-flags: -Znext-solver
//@ [with_clone] check-pass

//! Regression test for https://github.com/rust-lang/rust/issues/141241.
//! The conditional ToOwned implementation must explain its Clone requirement.

#![allow(dead_code)]

use std::borrow::Cow;

#[cfg_attr(with_clone, derive(Clone))]
struct A {}

struct B {
    test: Cow<'static, [A]>,
    //[current]~^ ERROR the trait bound `[A]: ToOwned` is not satisfied
    //[next]~^^ ERROR the trait bound `[A]: ToOwned` is not satisfied
}

fn main() {}
