//@ compile-flags: -Zvalidate-mir

// Regression test for <https://github.com/rust-lang/rust/issues/141394>.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::unwrap_binder;

fn id<T>(x: unsafe<> T) -> T {
    //~^ ERROR the trait bound `T: Copy` is not satisfied
    unsafe { unwrap_binder!(x) }
}

fn main() {}
