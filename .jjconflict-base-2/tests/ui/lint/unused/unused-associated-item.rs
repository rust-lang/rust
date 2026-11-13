//@ check-pass

#![deny(unused_must_use)]

use std::future::Future;
use std::pin::Pin;

trait Factory {
    type Output;
}

impl Factory for () {
    type Output = Pin<Box<dyn Future<Output = ()> + 'static>>;
}

// Make sure we don't get an `unused_must_use` error on the *associated type bound*.
fn f() -> impl Factory<Output: Future> {}

fn main() {
    f();
}
