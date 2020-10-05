// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]
#![allow(dead_code, unconditional_recursion)]

fn fact<const N: usize>() {
    fact::<{ N - 1 }>();
    //[full]~^ ERROR constant expression depends on a generic parameter
    //[min]~^^ ERROR generic parameters must not be used inside of non-trivial constant values
}

fn main() {}
