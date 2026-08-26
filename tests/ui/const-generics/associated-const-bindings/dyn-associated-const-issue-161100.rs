//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, min_generic_const_args)]

trait Trait {
    const F: fn();
}

fn take(_: &dyn Trait<F = { || {} }>) {}
//~^ ERROR using function pointers as const generic parameters is forbidden

fn main() {}
