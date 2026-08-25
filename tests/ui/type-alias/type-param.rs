//@ run-pass
// This is a smoke test to ensure that type aliases with type parameters
// are accepted by the compiler and that the parameters are correctly
// resolved in the aliased item type.

#![allow(dead_code)]

type Foo<T> = extern "C" fn(T) -> bool;
type Bar<T> = fn(T) -> bool;

fn main() {}
