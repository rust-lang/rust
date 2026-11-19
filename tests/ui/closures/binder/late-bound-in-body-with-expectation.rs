//@ check-pass
// Regression test for <https://github.com/rust-lang/rust/issues/141504>
// and <https://github.com/rust-lang/rust/issues/166121>.
// A closure with an explicit `for<...>` binder must keep its own lifetimes
// even when an expected signature is available.

#![feature(closure_lifetime_binder)]

use std::thread::{self, Scope, ScopedJoinHandle};

fn callback(_: impl for<'a> FnOnce(&'a mut i32)) {}
fn callback_no_args(_: impl FnOnce()) {}

fn main() {
    callback(for<'a> |y: &'a mut i32| -> () {
        let _: &'a mut i32 = y;
    });

    callback_no_args(for<'a> || -> () {
        let _: &'a i32 = &1;
    });

    let _: fn() = for<'a> || -> () {
        let _: &'a ();
    };

    thread::scope(for<'a, 'b> |s: &'a Scope<'a, 'b>| -> () {
        let _spawned: ScopedJoinHandle<'a, ()> = s.spawn(|| {});
    });
}
