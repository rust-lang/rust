//@ known-bug: #54888

#![feature(unsize, coerce_unsized)]

use std::{
    ops::CoerceUnsized,
    marker::Unsize,
};

#[repr(C)]
struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T>
where
    T: Unsize<U>,
{}


fn main() {
    let foo = Ptr(Box::new(5)) as Ptr<dyn std::any::Any>;
}
