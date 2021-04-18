// revisions: min
// FIXME(const_generics): add the `full` revision,
// currently causes an ICE as we don't supply substs to
// anon consts in the parameter listing, as that would
// cause that anon const to reference itself.
#![cfg_attr(full, feature(const_generics))]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

struct Foo<const N: usize, const M: usize = { N + 1 }>;
//[min]~^ ERROR generic parameters may not be used in const operations

struct Bar<T, const TYPE_SIZE: usize = { std::mem::size_of::<T>() }>(T);
//[min]~^ ERROR generic parameters may not be used in const operations
//[full]~^^ ERROR the size for values of type `T` cannot be known at compilation time

fn main() {}
