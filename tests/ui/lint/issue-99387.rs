// check-pass

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
    None.into_iter()
}

fn main() {}
