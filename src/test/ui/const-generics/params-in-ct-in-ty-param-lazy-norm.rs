#![feature(const_generics)] //~ WARN the feature `const_generics` is incomplete

struct Foo<T, U = [u8; std::mem::size_of::<T>()]>(T, U);
//~^ ERROR constant values inside of type parameter defaults

// FIXME(const_generics:defaults): We still don't know how to we deal with type defaults.
struct Bar<T = [u8; N], const N: usize>(T);
//~^ ERROR constant values inside of type parameter defaults
//~| ERROR type parameters with a default

fn main() {}
