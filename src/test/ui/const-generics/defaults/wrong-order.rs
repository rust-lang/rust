// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete

struct A<T = u32, const N: usize> {
    //~^ ERROR type parameters with a default must be trailing
    arg: T,
}

fn main() {}
