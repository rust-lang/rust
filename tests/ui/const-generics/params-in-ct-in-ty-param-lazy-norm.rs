//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo<T, U = [u8; std::mem::size_of::<T>()]>(T, U);
//[min]~^ ERROR generic parameters may not be used in const operations

struct Bar<T = [u8; N], const N: usize>(T);
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared
//~| ERROR generic parameters with a default

fn main() {}
