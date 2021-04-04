// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![allow(dead_code, unconditional_recursion)]

fn fact<const N: usize>() {
    fact::<{ N - 1 }>();
    //[full]~^ ERROR constant expression depends on a generic parameter
    //[min]~^^ ERROR generic parameters may not be used in const operations
}

fn main() {}
