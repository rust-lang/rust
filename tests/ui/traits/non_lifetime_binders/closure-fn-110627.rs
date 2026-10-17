//@ check-fail
//! Regression test for <https://github.com/rust-lang/rust/issues/110627>.
//!
//! Ensure type-checking a closure against a higher-ranked `for<T> Fn(T) -> T`
//! bound produces a type error instead of an ICE.

#![feature(non_lifetime_binders)]

fn take(_id: impl for<T> Fn(T) -> T) {}

fn main() {
    take(|x| x);
    //~^ ERROR type mismatch in closure arguments
}
