#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Bad<const N: usize, T> { //~ ERROR type parameters must be declared prior
    arr: [u8; { N }],
    another: T,
}

fn main() { }
