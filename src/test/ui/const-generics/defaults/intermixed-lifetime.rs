// revisions: full min
// Checks that lifetimes cannot be interspersed between consts and types.
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo<const N: usize, 'a, T = u32>(&'a (), T);
//~^ Error lifetime parameters must be declared prior to const parameters
//[min]~^^ Error type parameters must be declared prior to const parameters

struct Bar<const N: usize, T = u32, 'a>(&'a (), T);
//[full]~^ Error lifetime parameters must be declared prior to type parameters
//[min]~^^ Error type parameters must be declared prior to const parameters
//[min]~| Error lifetime parameters must be declared prior to const parameters

fn main() {}
