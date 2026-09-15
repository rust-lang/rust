//@ check-fail

#![feature(reborrow)]

use std::marker::Reborrow;

struct Thing;
impl<'a> Reborrow for Thing {}
//~^ ERROR implementing `Reborrow` does not allow multiple lifetimes or fields to be coerced
fn foo(_: Thing) {}
fn main() {
    let x = Thing;
    foo(x);
}
