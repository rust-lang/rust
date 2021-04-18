#![feature(const_generics, const_generics_defaults)]
#![allow(incomplete_features)]

struct Foo<const N: usize, const M: usize = { N + 1 }>;

struct Bar<T, const TYPE_SIZE: usize = { std::mem::size_of::<T>() }>(T);
//~^ ERROR the size for values of type `T` cannot be known at compilation time 

fn main() {}
