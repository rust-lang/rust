//! Test negative Sync implementation on structs.
//!
//! Uses the unstable `negative_impls` feature to explicitly opt-out of Sync.

#![feature(negative_impls)]

use std::marker::Sync;

struct NotSync {
    value: isize,
}

impl !Sync for NotSync {}

fn requires_sync<T: Sync>(_: T) {}

fn main() {
    let not_sync = NotSync { value: 5 };
    requires_sync(not_sync);
    //~^ ERROR `NotSync` cannot be shared between threads safely [E0277]
}
