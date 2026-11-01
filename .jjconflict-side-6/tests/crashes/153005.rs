//@ known-bug: #153005
#![feature(non_lifetime_binders)]
#![feature(derive_coerce_pointee)]

#[derive(core::marker::CoercePointee)]
#[repr(transparent)]
struct _Ptr5<'a, #[pointee] T: ?Sized, X>
where
    for<V> V: Sized,
{
    data: &'a T,
    x: core::marker::PhantomData<X>,
}

fn main() {}
