// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub type Successors<'a> = impl Iterator<Item = &'a ()>;

pub fn f<'a>() -> Successors<'a> {
    None.into_iter()
}

pub trait Tr {
    type Item;
}

impl<'a> Tr for &'a () {
    type Item = Successors<'a>;
}

pub fn kazusa<'a>() -> <&'a () as Tr>::Item {
    None.into_iter()
}

fn main() {}
