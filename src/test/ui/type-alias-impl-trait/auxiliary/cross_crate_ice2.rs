// Crate that exports an opaque `impl Trait` type. Used for testing cross-crate.

#![crate_type = "rlib"]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub trait View {
    type Tmp: Iterator<Item = u32>;

    fn test(&self) -> Self::Tmp;
}

pub struct X;

impl View for X {
    type Tmp = impl Iterator<Item = u32>;

    fn test(&self) -> Self::Tmp {
        vec![1,2,3].into_iter()
    }
}
