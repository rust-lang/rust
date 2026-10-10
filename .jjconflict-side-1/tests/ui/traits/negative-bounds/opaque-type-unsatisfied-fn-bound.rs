//@ compile-flags: -Znext-solver -Zinternal-testing-features

#![feature(negative_bounds, unboxed_closures)]

fn produce() -> impl !Fn<(u32,)> {}
//~^ ERROR the trait bound `(): !Fn(u32)` is not satisfied

fn main() {}
