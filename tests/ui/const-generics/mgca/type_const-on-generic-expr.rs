#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

#[type_const]
const FREE1<T>: usize = const { std::mem::size_of::<T>() };
//~^ ERROR generic parameters may not be used in const operations
#[type_const]
const FREE2<const I: usize>: usize = const { I + 1 };
//~^ ERROR generic parameters may not be used in const operations

fn main() {}
