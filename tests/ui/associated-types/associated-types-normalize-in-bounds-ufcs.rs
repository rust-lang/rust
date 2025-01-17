//@ check-pass
#![allow(unused_variables)]
// Test that we normalize associated types that appear in bounds; if
// we didn't, the call to `self.split2()` fails to type check.


use std::marker::PhantomData;

struct Splits<'a, T:'a, P>(PhantomData<(&'a T, P)>);
struct SplitsN<I>(PhantomData<I>);

trait SliceExt2 {
    type Item;

    fn split2<'a, P>(&'a self, pred: P) -> Splits<'a, Self::Item, P>
        where P: FnMut(&Self::Item) -> bool;
    fn splitn2<'a, P>(&'a self, n: u32, pred: P) -> SplitsN<Splits<'a, Self::Item, P>>
        where P: FnMut(&Self::Item) -> bool;
}

impl<T> SliceExt2 for [T] {
    type Item = T;

    fn split2<P>(&self, pred: P) -> Splits<T, P> where P: FnMut(&T) -> bool {
        loop {}
    }

    fn splitn2<P>(&self, n: u32, pred: P) -> SplitsN<Splits<T, P>> where P: FnMut(&T) -> bool {
        SliceExt2::split2(self, pred);
        loop {}
    }
}

fn main() { }
