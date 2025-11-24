#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

#[pin_v2]
struct Foo;
struct Bar;

impl Unpin for Foo {} //~ ERROR explicit impls for the `Unpin` trait are not permitted for structurally pinned types
impl Unpin for Bar {} // ok

fn main() {}
