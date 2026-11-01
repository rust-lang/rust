//! Test that we don't follow through projections to find
//! opaque types.

#![feature(type_alias_impl_trait)]
#![allow(private_interfaces)]

pub type Successors<'a> = impl Iterator<Item = &'a ()>;

#[define_opaque(Successors)]
pub fn f<'a>() -> Successors<'a> {
    None.into_iter()
}

trait Tr {
    type Item;
}

impl<'a> Tr for &'a () {
    type Item = Successors<'a>;
}

pub fn ohno<'a>() -> <&'a () as Tr>::Item {
    None.into_iter()
    //~^ ERROR mismatched types
}

fn main() {}
