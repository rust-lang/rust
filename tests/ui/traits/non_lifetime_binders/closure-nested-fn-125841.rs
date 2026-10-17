//@ check-fail
//! Regression test for <https://github.com/rust-lang/rust/issues/125841>.
//!
//! Ensure returning a closure for `impl for<T> Fn(T) -> T` and passing a closure
//! expected to satisfy `for<T> Fn(T) -> T` produces a type error instead of an ICE.

#![feature(non_lifetime_binders)]

fn take(_id: impl for<T> Fn(T) -> T) {}

fn take2() -> impl for<T> Fn(T) -> T {
    //~^ ERROR type mismatch in closure arguments
    |x| x
}

fn main() {
    take(|x| take2);
    //~^ ERROR type mismatch in closure arguments
}
