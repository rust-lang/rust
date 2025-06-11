//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(generic_const_items)]

trait Trait {
    const FAIL<'a, T>: &'a T;
    //~^ ERROR: unable to satisfy constraints because it cannot be proven
}

fn main() {}
