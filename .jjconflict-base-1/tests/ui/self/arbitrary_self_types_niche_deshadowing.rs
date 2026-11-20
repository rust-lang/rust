//@ run-pass

#![allow(dead_code)]
#![allow(incomplete_features)]

#![feature(arbitrary_self_types)]
#![feature(arbitrary_self_types_pointers)]
#![feature(pin_ergonomics)]

use std::pin::Pin;
use std::pin::pin;
use std::marker::PhantomData;

struct A;

impl A {
    fn m(self: *const SmartPtr<Self>) -> usize { 2 }
    fn n(self: *const SmartPtr<Self>) -> usize { 2 }

    fn o(self: Pin<&SmartPtr2<Self>>) -> usize { 2 }
    fn p(self: Pin<&SmartPtr2<Self>>) -> usize { 2 }
}

struct SmartPtr<T>(T);

impl<T> core::ops::Receiver for SmartPtr<T> {
    type Target = *mut T;
}

impl<T> SmartPtr<T> {
    // In general we try to detect cases where a method in a smart pointer
    // "shadows" a method in the referent (in this test, A).
    // This method "shadows" the 'n' method in the inner type A
    // We do not attempt to produce an error in these shadowing cases
    // since the type signature of this method and the corresponding
    // method in A are pretty unlikely to occur in practice,
    // and because it shows up conflicts between *const::cast and *mut::cast.
    fn n(self: *mut Self) -> usize { 1 }
}

struct SmartPtr2<'a, T>(T, PhantomData<&'a T>);

impl<'a, T> core::ops::Receiver for SmartPtr2<'a, T> {
    type Target = Pin<&'a mut T>;
}

impl<T> SmartPtr2<'_, T> {
    // Similarly, this method shadows the method in A
    // Can only happen with the unstable feature pin_ergonomics
    fn p(self: Pin<&mut Self>) -> usize { 1 }
}

fn main() {
    let mut sm = SmartPtr(A);
    let smp: *mut SmartPtr<A> = &mut sm as *mut SmartPtr<A>;
    assert_eq!(smp.m(), 2);
    assert_eq!(smp.n(), 1);

    let smp: Pin<&mut SmartPtr2<A>> = pin!(SmartPtr2(A, PhantomData));
    assert_eq!(smp.o(), 2);
    let smp: Pin<&mut SmartPtr2<A>> = pin!(SmartPtr2(A, PhantomData));
    assert_eq!(smp.p(), 1);
}
