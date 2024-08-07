//@ edition: 2021

#![feature(async_closure, target_feature_11)]
// `target_feature_11` just to test safe functions w/ target features.

use std::pin::Pin;
use std::future::Future;

unsafe extern "Rust" {
    pub unsafe fn unsafety() -> Pin<Box<dyn Future<Output = ()> + 'static>>;
}

unsafe extern "C" {
    pub safe fn abi() -> Pin<Box<dyn Future<Output = ()> + 'static>>;
}


#[target_feature(enable = "sse2")]
fn target_feature()  -> Pin<Box<dyn Future<Output = ()> + 'static>> { todo!() }

fn test(f: impl async Fn()) {}

fn main() {
    test(unsafety); //~ ERROR the trait bound
    test(abi); //~ ERROR the trait bound
    test(target_feature); //~ ERROR the trait bound
}
