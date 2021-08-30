#![feature(const_generics_defaults)]

struct A<T = u32, const N: usize> {
    //~^ ERROR generic parameters with a default must be trailing
    arg: T,
}

fn main() {}
