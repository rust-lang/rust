//@ compile-flags: -Znext-solver

#![feature(negative_bounds, unboxed_closures)]

fn produce() -> impl !Fn<(u32,)> {}
//~^ ERROR type mismatch resolving `impl !Fn<(u32,)> == ()`

fn main() {}
