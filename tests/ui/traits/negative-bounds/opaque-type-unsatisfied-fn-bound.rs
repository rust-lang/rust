//@ compile-flags: -Znext-solver

#![feature(negative_bounds, unboxed_closures)]

fn produce() -> impl !Fn<(u32,)> {}
//~^ ERROR the trait bound `(): !Fn(u32)` is not satisfied

fn main() {}
