//@ aux-build:unstable_impl_method_selection_aux.rs

extern crate unstable_impl_method_selection_aux as aux;
use aux::Trait;

// The test below should not infer the type based on the fact
// that `impl Trait for Vec<u64>` is unstable. This would cause breakage
// in downstream crate once `impl Trait for Vec<u64>` is stabilised.

fn bar() {
    vec![].foo();
    //~^ ERROR type annotations needed
}

fn main() {}
