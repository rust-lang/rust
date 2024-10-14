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
//~^ ERROR the size for values of type `()` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `()` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `impl !Sized + Sized` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `()` cannot be known at compilation time [E0277]
fn weird1() -> impl !Sized + Sized {}
//~^ ERROR the size for values of type `()` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `()` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `impl !Sized + Sized` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `()` cannot be known at compilation time [E0277]
fn weird2() -> impl !Sized {}
//~^ ERROR the size for values of type `()` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `()` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `impl !Sized` cannot be known at compilation time [E0277]
//~| ERROR the size for values of type `()` cannot be known at compilation time [E0277]
