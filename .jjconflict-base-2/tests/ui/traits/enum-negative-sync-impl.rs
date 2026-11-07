//! Test that enums inherit Sync/!Sync properties from their variants.
//!
//! Uses the unstable `negative_impls` feature to explicitly opt-out of Sync.

#![feature(negative_impls)]

use std::marker::Sync;

struct NoSync;
impl !Sync for NoSync {}

enum Container {
    WithNoSync(NoSync),
}

fn requires_sync<T: Sync>(_: T) {}

fn main() {
    let container = Container::WithNoSync(NoSync);
    requires_sync(container);
    //~^ ERROR `NoSync` cannot be shared between threads safely [E0277]
}
