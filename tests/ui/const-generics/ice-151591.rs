#![feature(adt_const_params)]
#![feature(unsized_const_params)]
//~^ WARN the feature `unsized_const_params` is incomplete and may not be safe to use and/or cause compiler crashes

#[derive(Clone)]
struct S<const L: [u8]>;

const A: [u8];
//~^ ERROR free constant item without body
//~| ERROR the size for values of type `[u8]` cannot be known at compilation time

impl<const N: i32> Copy for S<A> {}
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
//~| ERROR the const parameter `N` is not constrained by the impl trait, self type, or predicates
impl<const M: usize> Copy for S<A> {}
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
//~| ERROR the const parameter `M` is not constrained by the impl trait, self type, or predicates

fn main() {}
