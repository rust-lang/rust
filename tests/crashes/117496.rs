//@ known-bug: #117496
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

use core::marker::ConstParamTy;

#[derive(PartialEq, Copy, Clone, Eq, ConstParamTy)]
pub enum Foo {}
impl Foo {
    pub const fn size(self) -> usize {
        1
    }
}

pub struct Bar<const F: Foo, const SIZE: usize = { F.size() }>([u64; SIZE])
where
    [u64; SIZE]: Sized;

pub struct Quux<const F: Foo> {}
impl<const F: Foo> Quux<{ F }> {
    pub unsafe fn nothing(&self, bar: &mut Bar<{ F }>) {}
}
