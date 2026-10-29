//@ known-bug: #138564
//@compile-flags: -Copt-level=0 -Cdebuginfo=2 --crate-type lib
#![feature(unsize, dispatch_from_dyn, arbitrary_self_types)]

use std::marker::Unsize;
use std::ops::{Deref, DispatchFromDyn};

#[repr(align(16))]
pub struct MyPointer<T: ?Sized>(*const T);

impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<MyPointer<U>> for MyPointer<T> {}
impl<T: ?Sized> Deref for MyPointer<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unimplemented!()
    }
}

pub trait Trait {
    fn foo(self: MyPointer<Self>) {}
}

// make sure some usage of `<dyn Trait>::foo` makes it to codegen
pub fn user() -> *const () {
    <dyn Trait>::foo as *const ()
}
