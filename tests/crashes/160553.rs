//@ known-bug: #160553
//@ compile-flags: -Copt-level=0
#![allow(incomplete_features)]
#![feature(adt_const_params, min_generic_const_args, macroless_generic_const_args)]
#![feature(generic_const_parameter_types)]

trait Trait {
    type const LEN: usize;
}

struct S;
impl Trait for S {
    type const LEN: usize = 2;
}

fn foo<T: Trait, const A: [u8; <T as Trait>::LEN]>() -> [u8; <T as Trait>::LEN] {
    A
}

fn bar<T: Trait>() -> [u8; <T as Trait>::LEN] {
    foo::<T, { [1, 2, 3] }>()
}

fn main() {
    bar::<S>();
}
