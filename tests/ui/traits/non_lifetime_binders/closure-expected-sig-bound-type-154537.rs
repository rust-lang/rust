//@ check-fail
//! Regression test for <https://github.com/rust-lang/rust/issues/154537>.
//!
//! Avoid an ICE when closure expected signature contains non-lifetime
//! bound variables.

#![feature(non_lifetime_binders)]

fn take() -> impl for<T> Fn() -> T {
    //~^ ERROR type mismatch
    || 3
}

fn main() {}
