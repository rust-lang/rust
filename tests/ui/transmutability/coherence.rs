//! Regression test for <https://github.com/rust-lang/rust/pull/149228>

//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(transmutability)]
trait OpaqueTrait {}

impl<T: std::mem::TransmuteFrom<(), ()>> OpaqueTrait for T {}
//~^ ERROR: type provided when a constant was expected

impl<T> OpaqueTrait for &T where T: OpaqueTrait {}
//~^ ERROR: conflicting implementations of trait `OpaqueTrait` for type `&_`

fn main() {}
