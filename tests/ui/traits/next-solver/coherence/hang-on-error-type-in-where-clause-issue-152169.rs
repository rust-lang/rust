//! Regression test for <https://github.com/rust-lang/rust/issues/152169>.
//!
//! An unresolved type in the where clause of a blanket impl used to make
//! the overlap check hang.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Trait {}

impl<T> Trait for T where (Undefined, T): Trait {}
//~^ ERROR cannot find type `Undefined` in this scope

impl<T, W: Trait> Trait for (T, W) {}
//~^ ERROR conflicting implementations of trait `Trait` for type `(_, _)`

fn main() {}
