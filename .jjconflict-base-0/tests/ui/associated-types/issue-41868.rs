//@ check-pass

// Defaulted assoc. types should normalize properly in impls that don't
// override them.

#![feature(associated_type_defaults)]

pub struct Foo;

pub trait CanDecode: Sized {
    type Output = Self;
    fn read(rdr: &mut Foo) -> Option<Self::Output>;
}

impl CanDecode for u8 {
    fn read(rdr: &mut Foo) -> Option<Self::Output> { Some(42) }
}

impl CanDecode for u16 {
    fn read(rdr: &mut Foo) -> Option<u16> { Some(17) }
}

fn main() {}
