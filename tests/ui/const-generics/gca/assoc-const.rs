//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(min_generic_const_args, generic_const_args)]

trait Trait {
    const ASSOC: usize;
}

impl<T: Other> Trait for T {
    const ASSOC: usize = core::direct_const_arg!(T::RIGID);
}

trait Other {
    const RIGID: usize;
}

fn foo<T: Other>() {
    let a: [(); core::direct_const_arg!(<T as Trait>::ASSOC)] =
        [(); core::direct_const_arg!(T::RIGID)];
}

fn main() {}
