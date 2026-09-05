//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, min_generic_const_args)]
#![allow(incomplete_features)]

trait Trait {
    const F: fn();
}

trait Nested {
    type Out: Trait<F = { || {} }>;
    //~^ ERROR using function pointers as const generic parameters is forbidden
}

fn main() {}
