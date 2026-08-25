#![feature(const_trait_impl, min_specialization, rustc_attrs)]

#[rustc_specialization_trait]
pub const trait Sup {}

const impl Sup for () {}

pub const trait A {
    fn a() -> u32;
}

pub const trait Spec {}

const impl<T: [const] Spec> A for T {
    default fn a() -> u32 {
        2
    }
}

impl<T: Spec + Sup> A for T {
    //~^ ERROR conflicting implementations of trait `A`
    fn a() -> u32 {
        3
    }
}

fn main() {}
