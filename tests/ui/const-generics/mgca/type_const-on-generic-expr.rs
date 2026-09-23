#![expect(incomplete_features)]
#![feature(min_generic_const_args, generic_const_items)]

use std::gca;

const FREE1<T>: usize = gca!(const { std::mem::size_of::<T>() });
//~^ ERROR generic parameters may not be used in const operations

const FREE2<const I: usize>: usize = gca!(const { I + 1 });
//~^ ERROR generic parameters may not be used in const operations

fn main() {}
