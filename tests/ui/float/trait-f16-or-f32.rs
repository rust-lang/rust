//@ check-fail

#![feature(f16)]

trait Trait {}
impl Trait for f16 {}
impl Trait for f32 {}

fn foo(_: impl Trait) {}

fn main() {
    foo(1.0); //~ ERROR the trait bound `f64: Trait` is not satisfied
}
