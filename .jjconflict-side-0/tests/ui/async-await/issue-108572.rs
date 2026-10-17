//@ edition: 2021
//@ run-rustfix
#![allow(unused_must_use, dead_code)]

use std::future::Future;
fn foo() -> impl Future<Output=()> {
    async { }
}

fn bar(cx: &mut std::task::Context<'_>) {
    let fut = foo();
    fut.poll(cx);
    //~^ ERROR no method named `poll` found for opaque type `impl Future<Output = ()>` in the current scope [E0599]
}
fn main() {}
