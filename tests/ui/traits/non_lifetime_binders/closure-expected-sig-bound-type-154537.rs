//@ check-fail
//! Regression test for issues:
//! - <https://github.com/rust-lang/rust/issues/154537>
//! - <https://github.com/rust-lang/rust/issues/108814>
//! - <https://github.com/rust-lang/rust/issues/110627>
//! - <https://github.com/rust-lang/rust/issues/125841>
//!
//! FIXME(non_lifetime_binders): provide better diagnostics. Should non-lifetime binders
//! in opaques with fn traits bound be rejected?

#![feature(non_lifetime_binders)]

// Variant 1: #154537
fn test_return_fn() -> impl for<T> Fn() -> T {
    //~^ ERROR type mismatch
    || 3
}

// Variant 2: #108814
fn take_fnonce(_: impl for<T> FnOnce(T) -> T) {}

// Variant 3: #110627
fn take_fn(_id: impl for<T> Fn(T) -> T) {}

// Variant 4: #125841
fn take_nested_fn(_id: impl for<T> Fn(T) -> T) {}
fn take2() -> impl for<T> Fn(T) -> T {
    //~^ ERROR type mismatch in closure arguments
    |x| x
}

fn main() {
    take_fnonce(|x| x);
    //~^ ERROR type mismatch in closure arguments

    take_fn(|x| x);
    //~^ ERROR type mismatch in closure arguments

    take_nested_fn(|x| take2);
    //~^ ERROR type mismatch in closure arguments
}
