#![allow(stutter)]

#[warn(clippy::stutter)]
fn main() {}

#[warn(clippy::new_without_default_derive)]
struct Foo;

impl Foo {
    fn new() -> Self {
        Foo
    }
}
