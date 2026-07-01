//@ run-pass

#![allow(dead_code)]
#![allow(incomplete_features)]

#![feature(arbitrary_self_types)]
#![feature(pin_ergonomics)]

use std::pin::Pin;
use std::pin::pin;
use std::marker::PhantomData;

struct A;

impl A {
    fn o(self: Pin<&SmartPtr<Self>>) -> usize { 2 }
    fn p(self: Pin<&SmartPtr<Self>>) -> usize { 2 }
}

struct SmartPtr<'a, T>(T, PhantomData<&'a T>);

impl<'a, T> core::ops::Receiver for SmartPtr<'a, T> {
    type Target = Pin<&'a mut T>;
}

impl<T> SmartPtr<'_, T> {
    // We try to detect cases where a method in a smart pointer "shadows" a
    // method in the referent (in this test, A). This method "shadows" the 'p'
    // method in the inner type A. We do not attempt to produce an error in
    // these shadowing cases.
    // Can only happen with the unstable feature pin_ergonomics.
    fn p(self: Pin<&mut Self>) -> usize { 1 }
}

fn main() {
    let smp: Pin<&mut SmartPtr<A>> = pin!(SmartPtr(A, PhantomData));
    assert_eq!(smp.o(), 2);
    let smp: Pin<&mut SmartPtr<A>> = pin!(SmartPtr(A, PhantomData));
    assert_eq!(smp.p(), 1);
}
