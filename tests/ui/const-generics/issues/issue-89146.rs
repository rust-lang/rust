//@ build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub trait Foo {
    const SIZE: usize;

    fn to_bytes(&self) -> [u8; Self::SIZE];
}

pub fn bar<G: Foo>(a: &G) -> u8
where
    [(); G::SIZE]: Sized,
{
    deeper_bar(a)
}

fn deeper_bar<G: Foo>(a: &G) -> u8
where
    [(); G::SIZE]: Sized,
{
    a.to_bytes()[0]
}

fn main() {}
