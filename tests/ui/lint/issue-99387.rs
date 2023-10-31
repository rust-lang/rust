//! Test that we don't follow through projections to find
//! opaque types.

#![feature(type_alias_impl_trait)]
#![allow(private_in_public)]

pub type Successors<'a> = impl Iterator<Item = &'a ()>;

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
    //~^ ERROR item constrains opaque type that is not in its signature
    None.into_iter()
}

fn main() {}
