//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(min_generic_const_args, generic_const_args)]

use std::gca;

trait Trait {
    const ASSOC: usize;
}

impl<T: Other> Trait for T {
    const ASSOC: usize = gca!(T::RIGID);
}

trait Other {
    const RIGID: usize;
}

fn foo<T: Other>() {
    let a: [(); gca!(<T as Trait>::ASSOC)] = [(); gca!(T::RIGID)];
}

fn main() {}
