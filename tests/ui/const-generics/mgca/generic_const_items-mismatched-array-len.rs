//! Regression test for #160553 (used to ICE)
#![allow(incomplete_features)]
#![feature(adt_const_params, gca_min_const_items, gca_macroless_args)]
#![feature(generic_const_parameter_types)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const LEN: usize;
}

struct S;
impl Trait for S {
    const LEN: usize = gca!(2);
}

fn foo<T: Trait, const A: [u8; <T as Trait>::LEN]>() -> [u8; <T as Trait>::LEN] {
    A
}

fn bar<T: Trait>() -> [u8; <T as Trait>::LEN] {
    foo::<T, { [1, 2, 3] }>()
    //~^ ERROR the constant `*b"\x01\x02\x03"` is not of type `[u8; <T as Trait>::LEN]`
}

fn main() {
    bar::<S>();
}
