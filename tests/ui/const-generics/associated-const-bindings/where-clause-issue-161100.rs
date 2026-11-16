//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items, gca_min_const_items)]

trait Trait {
    const F: fn();
}

fn take<T>() where T: Trait<F = { || {} }> {}
//~^ ERROR using function pointers as const generic parameters is forbidden

fn main() {}
