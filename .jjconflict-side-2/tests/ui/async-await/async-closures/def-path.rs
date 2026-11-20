//@ compile-flags: -Zverbose-internals
//@ edition:2021

fn main() {
    let x = async || {};
    //~^ NOTE the expected `async` closure body
    let () = x();
    //~^ ERROR mismatched types
    //~| NOTE this expression has type `{static main::{closure#0}::{closure#0}<
    //~| NOTE expected `async` closure body, found `()`
    //~| NOTE expected `async` closure body `{static main::{closure#0}::{closure#0}<
}
