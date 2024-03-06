//@ run-rustfix
#![allow(dead_code)]

#[derive(Clone)]
struct Wrapper<T>(T);

impl<S> Copy for Wrapper<S> {}
//~^ ERROR the trait `Copy` cannot be implemented for this type

fn main() {}
