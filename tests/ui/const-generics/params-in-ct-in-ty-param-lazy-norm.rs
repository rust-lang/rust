//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]

// FIXME(fmease): Don't emit additional+wrong diagnostics about "implicit forward references".

struct Foo<T, U = [u8; std::mem::size_of::<T>()]>(T, U);
//[min]~^ ERROR generic parameters may not be used in const operations
//[full]~^^ ERROR type parameter default references generic parameter `U` before it is declared

struct Bar<T = [u8; N], const N: usize>(T);
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared
//~| ERROR generic parameters with a default
//[full]~^^^ ERROR type parameter default references generic parameter `T` before it is declared
//[full]~| ERROR type parameter default references generic parameter `N` before it is declared

fn main() {}
