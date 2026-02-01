// Regression test for #117496: ConstParamTy and default const in Bar used to ICE in ArgFolder.

#![allow(incomplete_features)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

use core::marker::ConstParamTy;
//~^ ERROR failed to resolve: you might be missing crate `core`

#[derive(PartialEq, Copy, Clone, Eq, ConstParamTy)]
pub enum Foo {}
impl Foo {
    pub const fn size(self) -> usize {
        1
    }
}

pub struct Bar<const F: Foo, const SIZE: usize = { F.size() }>([u64; SIZE]) //~ ERROR `Foo` must implement `ConstParamTy`
where
    [u64; SIZE]: Sized;

pub struct Quux<const F: Foo> {}
//~^ ERROR `Foo` must implement `ConstParamTy`
impl<const F: Foo> Quux<{ F }> {
    //~^ ERROR `Foo` must implement `ConstParamTy`
    pub unsafe fn nothing(&self, bar: &mut Bar<{ F }>) {}
    //~^ ERROR unconstrained generic constant
}

fn main() {}
