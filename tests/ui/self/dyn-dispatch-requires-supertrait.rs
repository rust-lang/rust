//@ check-pass

#![feature(derive_coerce_pointee)]
#![feature(arbitrary_self_types)]

use std::ops::Deref;
use std::marker::CoercePointee;
use std::sync::Arc;

trait MyTrait {}

#[derive(CoercePointee)]
#[repr(transparent)]
struct MyArc<T>
where
    T: MyTrait + ?Sized,
{
    inner: Arc<T>
}

impl<T: MyTrait + ?Sized> Deref for MyArc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

// Proving that `MyArc<Self>` is dyn-dispatchable requires proving `MyArc<T>` implements
// `DispatchFromDyn<MyArc<U>>`. The `DispatchFromDyn` impl that is generated from the
// `CoercePointee` implementation requires the pointee impls `MyTrait`, but previously we
// were only assuming the pointee impl'd `MyOtherTrait`. Elaboration comes to the rescue here.
trait MyOtherTrait: MyTrait {
    fn foo(self: MyArc<Self>);
}

fn test(_: MyArc<dyn MyOtherTrait>) {}

fn main() {}
