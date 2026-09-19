#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

const FREE1<T>: usize = core::direct_const_arg!(const { std::mem::size_of::<T>() });
//~^ ERROR generic parameters may not be used in const operations

const FREE2<const I: usize>: usize = core::direct_const_arg!(const { I + 1 });
//~^ ERROR generic parameters may not be used in const operations

fn main() {}
