// Checks that lifetimes cannot be interspersed between consts and types.

struct Foo<const N: usize, 'a, T = u32>(&'a (), T);
//~^ ERROR lifetime parameters must be declared prior to type and const parameters

struct Bar<const N: usize, T = u32, 'a>(&'a (), T);
//~^ ERROR lifetime parameters must be declared prior to type and const parameters

fn main() {}
