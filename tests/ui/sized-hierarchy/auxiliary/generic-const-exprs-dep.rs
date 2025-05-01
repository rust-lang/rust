#![feature(generic_const_exprs)]

pub trait Tr<PCS, const C: usize> {
    fn foo(bar: PCS) { unimplemented!(); }
}

pub struct Foo;

pub const M: usize = 4;

impl<CS> Tr<CS, M> for Foo
{
}
