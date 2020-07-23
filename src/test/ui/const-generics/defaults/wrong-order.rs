#![feature(const_generics)] //~ WARN the feature `const_generics` is incomplete

struct A<T = u32, const N: usize> {
    //~^ ERROR type parameters with a default must be trailing
    arg: T,
}

fn main() {}
