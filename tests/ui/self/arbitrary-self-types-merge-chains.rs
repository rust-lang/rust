// gate-test-arbitrary_self_types_merge_chains

#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_merge_chains)]

use std::marker::PhantomData;
use std::ops::{Deref, Receiver};

struct A;
impl Deref for A {
    type Target = u8;
    fn deref(&self) -> &u8 {
        &0
    }
}
impl Receiver for A {
    //~^ ERROR `Deref::Target` does not agree with `Receiver::Target`
    type Target = u32;
}

struct B<'a, T>(&'a T);
impl<'a, T> Deref for B<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &*self.0
    }
}
impl<'a, T> Receiver for B<'a, T> {
    //~^ ERROR `Deref::Target` does not agree with `Receiver::Target`
    type Target = Box<T>;
}

struct C<'a, T>(&'a T);
impl<'a, T> Deref for C<'a, T> {
    type Target = Self;
    fn deref(&self) -> &Self {
        self
    }
}
impl<'b, T> Receiver for C<'b, T> {
    type Target = Self; // OK
}

struct D<T>(PhantomData<fn() -> T>);
trait Trait {
    type Output;
}
impl<T: Trait<Output = T>> Deref for D<T> {
    type Target = T::Output;
    fn deref(&self) -> &T {
        unimplemented!()
    }
}
impl<T> Receiver for D<T> {
    type Target = T; // OK
}

fn main() {}
