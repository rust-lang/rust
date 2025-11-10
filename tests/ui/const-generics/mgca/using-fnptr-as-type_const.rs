// Regression test for #119783

#![expect(incomplete_features)]
#![feature(associated_const_equality, min_generic_const_args)]

trait Trait {
    #[type_const]
    const F: fn();
    //~^ ERROR using function pointers as const generic parameters is forbidden
}

fn take(_: impl Trait<F = { || {} }>) {}

fn main() {}
