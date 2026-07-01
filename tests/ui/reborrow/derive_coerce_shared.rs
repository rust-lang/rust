//@ run-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomRef<'a, T>(&'a T);
impl<'a, T> Clone for CustomRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for CustomRef<'a, T> {}

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(CustomRef<'a, T>)]
struct CustomMut<'a, T> {
    value: &'a mut T,
}

struct TupleRef<'a, T>(&'a T);
impl<'a, T> Clone for TupleRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for TupleRef<'a, T> {}

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(TupleRef<'a, T>)]
struct TupleMut<'a, T>(&'a mut T);

struct ArrayRef<'a, T, const N: usize>(&'a [T; N]);
impl<'a, T, const N: usize> Clone for ArrayRef<'a, T, N> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T, const N: usize> Copy for ArrayRef<'a, T, N> {}

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(ArrayRef<'a, T, N>)]
struct ArrayMut<'a, T, const N: usize>(&'a mut [T; N]);

#[derive(Clone, Copy)]
struct MarkerRef<'a>(PhantomData<&'a ()>);

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(MarkerRef<'a>)]
struct MarkerMut<'a>(PhantomData<&'a ()>);

#[derive(Clone, Copy)]
struct SharedOnlyRef<'a>(&'a ());

#[derive(CoerceShared)]
#[coerce_shared(SharedOnlyRef<'a>)]
struct SharedOnlyMut<'a>(&'a mut ());
impl<'a> Reborrow for SharedOnlyMut<'a> {}

fn take_custom(_: CustomRef<'_, ()>) {}
fn take_tuple(_: TupleRef<'_, ()>) {}
fn take_array(_: ArrayRef<'_, (), 1>) {}
fn take_marker<'a>(_: MarkerRef<'a>) -> &'a () {
    &()
}
fn take_shared_only(_: SharedOnlyRef<'_>) {}

fn main() {
    let custom = CustomMut { value: &mut () };
    take_custom(custom);

    let tuple = TupleMut(&mut ());
    take_tuple(tuple);

    let array = ArrayMut(&mut [()]);
    take_array(array);

    let marker = MarkerMut(PhantomData);
    let _ = take_marker(marker);
    let _ = take_marker(marker);

    let shared_only = SharedOnlyMut(&mut ());
    take_shared_only(shared_only);
}
