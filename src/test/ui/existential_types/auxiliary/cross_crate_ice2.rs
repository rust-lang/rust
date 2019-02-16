// Crate that exports an existential type. Used for testing cross-crate.

#![crate_type="rlib"]

#![feature(existential_type)]

pub trait View {
    type Tmp: Iterator<Item = u32>;

    fn test(&self) -> Self::Tmp;
}

pub struct X;

impl View for X {
    existential type Tmp: Iterator<Item = u32>;

    fn test(&self) -> Self::Tmp {
        vec![1,2,3].into_iter()
    }
}
