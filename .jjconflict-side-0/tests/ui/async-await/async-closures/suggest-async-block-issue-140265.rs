//@ edition:2024
// Test that we suggest using `async {}` block instead of `async || {}` closure if possible

use std::future::Future;

fn takes_future(_fut: impl Future<Output = ()>) {}

fn main() {
    // Basic case: suggest using async block
    takes_future(async || {
        //~^ ERROR is not a future
        println!("hi!");
    });

    // Without space between `||` and `{`: should also suggest using async block
    takes_future(async||{
        //~^ ERROR is not a future
        println!("no space!");
    });

    // With arguments: should suggest calling the closure, not using async block
    takes_future(async |x: i32| {
        //~^ ERROR is not a future
        println!("{x}");
    });
}
