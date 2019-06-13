#![allow(stutter)]
#![warn(clippy::cyclomatic_complexity)]

#[warn(clippy::stutter)]
fn main() {}

#[warn(clippy::new_without_default_derive)]
struct Foo;

#[warn(clippy::const_static_lifetime)]
static Bar: &'static str = "baz";

impl Foo {
    fn new() -> Self {
        Foo
    }
}
