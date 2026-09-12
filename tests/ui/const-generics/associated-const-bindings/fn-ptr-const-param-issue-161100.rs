//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args)]
#![feature(min_generic_const_args)]

trait Trait {
    const F: fn();
}

fn take(_: impl Trait<F = { || {} }>) {}
//~^ ERROR using function pointers as const generic parameters is forbidden

fn main() {}
