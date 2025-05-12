//@ revisions: rpass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
pub trait IsTrue {}
pub trait IsFalse {}

pub struct Assert<const CHECK: bool> {}

impl IsTrue for Assert<true> {}
impl IsFalse for Assert<false> {}

pub struct SliceConstWriter<'a, const N: usize> {
    ptr: &'a mut [u8],
}
impl<'a, const N: usize> SliceConstWriter<'a, { N }> {
    pub fn from_slice(vec: &'a mut [u8]) -> Self {
        Self { ptr: vec }
    }

    pub fn convert<const NN: usize>(mut self) -> SliceConstWriter<'a, { NN }> {
        SliceConstWriter { ptr: self.ptr }
    }
}

impl<'a, const N: usize> SliceConstWriter<'a, { N }>
where
    Assert<{ N >= 2 }>: IsTrue,
{
    pub fn write_u8(mut self) -> SliceConstWriter<'a, { N - 2 }> {
        self.convert()
    }
}

fn main() {}
