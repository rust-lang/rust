//@ check-fail
//! Regression test for <https://github.com/rust-lang/rust/issues/108814>.
//!
//! Ensure type-checking a closure against a higher-ranked `for<T> FnOnce(T) -> T`
//! bound produces a type error instead of an ICE.

#![feature(non_lifetime_binders)]

fn take(_: impl for<T> FnOnce(T) -> T) {}

fn main() {
    take(|x| x);
    //~^ ERROR type mismatch in closure arguments
}
