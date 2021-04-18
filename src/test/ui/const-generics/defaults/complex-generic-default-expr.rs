// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

struct Foo<const N: usize, const M: usize = { N + 1 }>;
//[min]~^ ERROR generic parameters may not be used in const operations

struct Bar<T, const TYPE_SIZE: usize = { std::mem::size_of::<T>() }>(T);
//[min]~^ ERROR generic parameters may not be used in const operations
//[full]~^^ ERROR the size for values of type `T` cannot be known at compilation time 

fn main() {}
