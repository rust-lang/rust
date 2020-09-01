// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete
#![cfg_attr(min, feature(min_const_generics))]

struct A<T = u32, const N: usize> {
    //~^ ERROR type parameters with a default must be trailing
    arg: T,
}

fn main() {}
