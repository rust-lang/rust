//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items)]
#![feature(gca_min_const_items)]

trait Trait {
    const F: fn();
}

fn take(_: impl Trait<F = { || {} }>) {}
//~^ ERROR using function pointers as const generic parameters is forbidden

fn main() {}
