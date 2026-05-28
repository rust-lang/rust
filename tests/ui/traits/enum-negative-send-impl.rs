//! Test that enums inherit Send/!Send properties from their variants.
//!
//! Uses the unstable `negative_impls` feature to explicitly opt-out of Send.

#![feature(negative_impls)]

use std::marker::Send;

struct NoSend;
impl !Send for NoSend {}

enum Container {
    WithNoSend(NoSend),
}

fn requires_send<T: Send>(_: T) {}

fn main() {
    let container = Container::WithNoSend(NoSend);
    requires_send(container);
    //~^ ERROR `NoSend` cannot be sent between threads safely
}
