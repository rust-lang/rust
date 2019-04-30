#![feature(unsize, dispatch_from_dyn, never_type)]

#![allow(dead_code)]

use std::{
    ops::DispatchFromDyn,
    marker::{Unsize, PhantomData},
};

struct Zst;
struct NestedZst(PhantomData<()>, Zst);


struct WithUnit<T: ?Sized>(Box<T>, ());
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<WithUnit<U>> for WithUnit<T>
    where T: Unsize<U> {}

struct WithPhantom<T: ?Sized>(Box<T>, PhantomData<()>);
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<WithPhantom<U>> for WithPhantom<T>
    where T: Unsize<U> {}

struct WithNever<T: ?Sized>(Box<T>, !);
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<WithNever<U>> for WithNever<T>
    where T: Unsize<U> {}

struct WithZst<T: ?Sized>(Box<T>, Zst);
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<WithZst<U>> for WithZst<T>
    where T: Unsize<U> {}

struct WithNestedZst<T: ?Sized>(Box<T>, NestedZst);
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<WithNestedZst<U>> for WithNestedZst<T>
    where T: Unsize<U> {}


struct Generic<T: ?Sized, A>(Box<T>, A);
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Generic<U, ()>> for Generic<T, ()>
    where T: Unsize<U> {}
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Generic<U, PhantomData<()>>>
    for Generic<T, PhantomData<()>>
    where T: Unsize<U> {}
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Generic<U, !>> for Generic<T, !>
    where T: Unsize<U> {}
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Generic<U, Zst>> for Generic<T, Zst>
    where T: Unsize<U> {}
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Generic<U, NestedZst>> for Generic<T, NestedZst>
    where T: Unsize<U> {}


fn main() {}
