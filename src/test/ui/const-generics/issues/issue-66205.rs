#![allow(incomplete_features, dead_code, unconditional_recursion)]
#![feature(const_generics)]

fn fact<const N: usize>() {
    fact::<{ N - 1 }>();
    //~^ ERROR constant expression depends on a generic parameter
}

fn main() {}
