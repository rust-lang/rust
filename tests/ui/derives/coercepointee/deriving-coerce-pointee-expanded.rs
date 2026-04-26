//@ check-pass
//@ compile-flags: -Zunpretty=expanded
//@ edition: 2015
#![feature(derive_coerce_pointee)]
use std::marker::CoercePointee;

pub trait MyTrait<T: ?Sized> {}

#[derive(CoercePointee)]
#[repr(transparent)]
struct MyPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct MyPointer2<'a, Y, Z: MyTrait<T>, #[pointee] T: ?Sized + MyTrait<T>, X: MyTrait<T> = ()>
where
    Y: MyTrait<T>,
{
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct MyPointerWithoutPointee<'a, T: ?Sized> {
    ptr: &'a T,
}
