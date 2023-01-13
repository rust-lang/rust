#![feature(const_trait_impl, min_specialization, rustc_attrs)]

#[rustc_specialization_trait]
#[const_trait]
pub trait Sup {}

impl const Sup for () {}

#[const_trait]
pub trait A {
    fn a() -> u32;
}

impl<T: Default> A for T {
    default fn a() -> u32 {
        2
    }
}

impl<T: Default + ~const Sup> const A for T {
    fn a() -> u32 {
        3
    }
}

const fn generic<T: Default>() {
    <T as A>::a();
    //~^ ERROR: the trait bound `T: ~const Sup` is not satisfied
}

fn main() {}
