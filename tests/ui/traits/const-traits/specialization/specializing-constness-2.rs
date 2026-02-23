#![feature(const_trait_impl, min_specialization, rustc_attrs)]
#![allow(internal_features)]

#[rustc_specialization_trait]
pub const trait Sup {}

impl const Sup for () {}

pub const trait A {
    fn a() -> u32;
}

impl<T: Default> A for T {
    default fn a() -> u32 {
        2
    }
}

impl<T: Default + [const] Sup> const A for T {
    fn a() -> u32 {
        3
    }
}

const fn generic<T: Default>() {
    <T as A>::a();
    //~^ ERROR: the trait bound `T: [const] A` is not satisfied
}

fn main() {}
