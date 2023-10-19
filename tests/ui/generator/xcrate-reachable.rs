// run-pass

// aux-build:xcrate-reachable.rs

#![feature(generator_trait)]

extern crate xcrate_reachable as foo;

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    Pin::new(&mut foo::foo()).resume(());
}
