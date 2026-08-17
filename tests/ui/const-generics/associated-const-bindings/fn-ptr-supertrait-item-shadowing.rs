//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args)]
#![feature(min_generic_const_args)]
#![feature(supertrait_item_shadowing)]

trait Super {
    const F: fn();
}

trait Sub: Super {
    const F: fn();
}

trait Leaf: Sub {}

fn take(_: impl Leaf<F = { || {} }>) {}
//~^ ERROR

fn main() {}
