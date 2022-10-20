#![feature(const_trait_impl, min_specialization, rustc_attrs)]
#![feature(effects)]

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

impl<T: Default + ~const Sup> const A for T { //~ ERROR: conflicting impl
    fn a() -> u32 {
        3
    }
}

const fn generic<T: Default>() {
    <T as A>::a();
}

fn main() {}
