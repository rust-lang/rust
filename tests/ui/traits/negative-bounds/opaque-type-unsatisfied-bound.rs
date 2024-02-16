//@ compile-flags: -Znext-solver

#![feature(negative_bounds, negative_impls)]

trait Trait {}
impl !Trait for () {}

fn produce() -> impl !Trait {}
fn consume(_: impl Trait) {}

fn main() {
    consume(produce()); //~ ERROR the trait bound `impl !Trait: Trait` is not satisfied
}

fn weird0() -> impl Sized + !Sized {}
//~^ ERROR type mismatch resolving `() == impl !Sized + Sized`
fn weird1() -> impl !Sized + Sized {}
//~^ ERROR type mismatch resolving `() == impl !Sized + Sized`
fn weird2() -> impl !Sized {}
//~^ ERROR type mismatch resolving `() == impl !Sized`
