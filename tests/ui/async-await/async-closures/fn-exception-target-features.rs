//@ edition: 2021
//@ only-x86_64

#![feature(async_closure, target_feature_11)]
// `target_feature_11` just to test safe functions w/ target features.

use std::pin::Pin;
use std::future::Future;

#[target_feature(enable = "sse2")]
fn target_feature()  -> Pin<Box<dyn Future<Output = ()> + 'static>> { todo!() }

fn test(f: impl AsyncFn()) {}

fn main() {
    test(target_feature); //~ ERROR the trait bound
}
