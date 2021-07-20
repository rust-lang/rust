// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub type Successors<'a> = impl Iterator<Item = &'a ()>;

pub fn f<'b>() -> Successors<'b> {
    None.into_iter()
}

pub trait Tr {
    type Item;
}

impl<'c> Tr for &'c () {
    type Item = Successors<'c>;
}

pub fn kazusa<'d>() -> <&'d () as Tr>::Item {
    None.into_iter()
}

fn main() {}
