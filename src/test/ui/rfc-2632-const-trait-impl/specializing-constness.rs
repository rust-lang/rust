#![feature(const_trait_impl, min_specialization, rustc_attrs)]

#[rustc_specialization_trait]
#[const_trait]
pub trait Sup {}

impl const Sup for () {}

#[const_trait]
pub trait A {
    fn a() -> u32;
}

impl<T: ~const Default> const A for T {
    default fn a() -> u32 {
        2
    }
}

impl<T: Default + Sup> A for T { //~ ERROR: cannot specialize
    fn a() -> u32 {
        3
    }
}

fn main() {}
