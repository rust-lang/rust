//@ compile-flags: -Znext-solver

#![feature(negative_bounds, unboxed_closures)]

fn produce() -> impl !Fn<(u32,)> {}
//~^ ERROR type mismatch resolving

fn main() {}
