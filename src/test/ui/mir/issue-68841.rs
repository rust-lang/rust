// compile-flags: -Z mir-opt-level=2
// edition:2018
// build-pass

#![feature(async_closure)]

use std::future::Future;

fn async_closure() -> impl Future<Output = u8> {
    (async move || -> u8 { 42 })()
}

fn main() {
    let _fut = async_closure();
}
