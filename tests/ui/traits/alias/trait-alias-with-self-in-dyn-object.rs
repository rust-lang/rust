//! Regression test for https://github.com/rust-lang/rust/issues/138891.

#![feature(trait_alias)]
trait F = Fn() -> Self;

fn _f3<Fut>(a: dyn F<Fut>) {}
//~^ ERROR trait alias takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR associated type binding in trait object type mentions `Self`

fn main() {}
