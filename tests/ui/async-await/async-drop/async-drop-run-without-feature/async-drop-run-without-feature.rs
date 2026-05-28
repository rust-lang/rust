//@ edition: 2024
//@ run-pass
//@ compile-flags: -Z mir-opt-level=0
//@ aux-crate: async_drop_crate_dep=async-drop-crate-dep.rs

use std::{ //~ WARN found async drop types in dependency
    pin::pin,
    task::{Context, Waker},
};

extern crate async_drop_crate_dep;

fn main() {
    let mut context = Context::from_waker(Waker::noop());
    let future = pin!(async { async_drop_crate_dep::run().await });
    // For some reason, putting this value into a variable is load-bearing.
    let _x = future.poll(&mut context);
}
