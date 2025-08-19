// Check that a self parameter type requires a DispatchFromDyn impl to be dyn-compatible.

#![feature(arbitrary_self_types, unsize, coerce_unsized)]

use std::{
    marker::Unsize,
    ops::{CoerceUnsized, Deref},
};

struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

impl<T: Unsize<U> + ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T> {}
// Because this impl is missing the coercion below fails.
// impl<T: Unsize<U> + ?Sized, U: ?Sized> DispatchFromDyn<Ptr<U>> for Ptr<T> {}

trait Trait {
    fn ptr(self: Ptr<Self>);
}
impl Trait for i32 {
    fn ptr(self: Ptr<Self>) {}
}

fn main() {
    Ptr(Box::new(4)) as Ptr<dyn Trait>;
    //~^ ERROR the trait `Trait` is not dyn compatible
}
