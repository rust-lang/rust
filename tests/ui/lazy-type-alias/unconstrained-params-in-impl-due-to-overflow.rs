#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

impl<T> Loop<T> {} //~ ERROR the type parameter `T` is not constrained

type Loop<T> = Loop<T>; //~ ERROR overflow

fn main() {}
