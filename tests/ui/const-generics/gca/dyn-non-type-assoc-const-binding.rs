//@ check-pass
//@ compile-flags: -Znext-solver=globally

#![feature(gca_min_const_items, gca_const_items)]
#![expect(incomplete_features)]

trait Trait {
    const ASSOC: usize;
}

fn foo(_: &dyn Trait<ASSOC = 10>) {}

fn main() {}
