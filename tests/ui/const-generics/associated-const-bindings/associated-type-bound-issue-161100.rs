//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items, gca_min_const_items)]
#![allow(incomplete_features)]

trait Trait {
    const F: fn();
}

trait Nested {
    type Out: Trait<F = { || {} }>;
    //~^ ERROR using function pointers as const generic parameters is forbidden
}

fn main() {}
