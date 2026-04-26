//@ check-pass

#![feature(derive_coerce_pointee)]

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr<'a, #[pointee] T: OnDrop + ?Sized, X> {
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

pub trait OnDrop {
    fn on_drop(&mut self);
}

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr2<'a, #[pointee] T: ?Sized, X>
where
    T: OnDrop,
{
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

pub trait MyTrait<T: ?Sized> {}

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr3<'a, #[pointee] T: ?Sized, X>
where
    T: MyTrait<T>,
{
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr4<'a, #[pointee] T: MyTrait<T> + ?Sized, X> {
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr5<'a, #[pointee] T: ?Sized, X>
where
    Ptr5Companion<T>: MyTrait<T>,
    Ptr5Companion2: MyTrait<T>,
{
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

pub struct Ptr5Companion<T: ?Sized>(core::marker::PhantomData<T>);
pub struct Ptr5Companion2;

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
pub struct Ptr6<'a, #[pointee] T: ?Sized, X: MyTrait<T> = (), const PARAM: usize = 0> {
    data: &'a mut T,
    x: core::marker::PhantomData<X>,
}

// a reduced example from https://lore.kernel.org/all/20240402-linked-list-v1-1-b1c59ba7ae3b@google.com/
#[repr(transparent)]
#[derive(core::marker::CoercePointee)]
pub struct ListArc<#[pointee] T, const ID: u64 = 0>
where
    T: ListArcSafe<ID> + ?Sized,
{
    arc: *const T,
}

pub trait ListArcSafe<const ID: u64> {}

fn main() {}
