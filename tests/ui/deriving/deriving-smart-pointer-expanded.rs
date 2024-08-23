//@ check-pass
//@ compile-flags: -Zunpretty=expanded
#![feature(derive_smart_pointer)]
use std::marker::SmartPointer;

pub trait MyTrait<T: ?Sized> {}

#[derive(SmartPointer)]
#[repr(transparent)]
struct MyPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

#[derive(core::marker::SmartPointer)]
#[repr(transparent)]
pub struct MyPointer2<'a, Y, Z: MyTrait<T>, #[pointee] T: ?Sized + MyTrait<T>, X: MyTrait<T> = ()>
where
    Y: MyTrait<T>,
{
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

#[derive(SmartPointer)]
#[repr(transparent)]
struct MyPointerWithoutPointee<'a, T: ?Sized> {
    ptr: &'a T,
}
