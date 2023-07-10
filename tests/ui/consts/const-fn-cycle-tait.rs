#![feature(type_alias_impl_trait)]

//! Test that we pessimistically assume the `Drop` impl of
//! a hidden type is not const.

pub struct Parser<H>(H);

type Tait = impl Sized;

const fn constrain() -> Tait {}

pub const fn take(_: Tait) {}
//~^ ERROR: destructor of `Tait` cannot be evaluated at compile-time

fn main() {
    println!("Hello, world!");
}
