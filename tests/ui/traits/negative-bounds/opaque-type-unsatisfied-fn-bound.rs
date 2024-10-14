//@ compile-flags: -Znext-solver

#![feature(negative_bounds, unboxed_closures)]

fn produce() -> impl !Fn<(u32,)> {}
//~^ ERROR expected a `Fn(u32)` closure, found `()`
//~| ERROR expected a `Fn(u32)` closure, found `()`
//~| ERROR the size for values of type `impl !Fn<(u32,)>` cannot be known at compilation time
//~| ERROR expected a `Fn(u32)` closure, found `()`

fn main() {}
