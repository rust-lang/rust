//@ run-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct SourceLeadingMut<'a, T>(PhantomData<&'a mut T>, &'a mut T);

impl<'a, T> Reborrow for SourceLeadingMut<'a, T> {}

struct SourceLeadingRef<'a, T>(&'a T);

impl<'a, T> Clone for SourceLeadingRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for SourceLeadingRef<'a, T> {}

impl<'a, T> CoerceShared<SourceLeadingRef<'a, T>> for SourceLeadingMut<'a, T> {}

struct TargetLeadingMut<'a, T>(&'a mut T);

impl<'a, T> Reborrow for TargetLeadingMut<'a, T> {}

struct TargetLeadingRef<'a, T>(PhantomData<&'a T>, &'a T);

impl<'a, T> Clone for TargetLeadingRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for TargetLeadingRef<'a, T> {}

impl<'a, T> CoerceShared<TargetLeadingRef<'a, T>> for TargetLeadingMut<'a, T> {}

struct InterleavedMut<'a, T, U>(
    PhantomData<&'a mut T>,
    &'a mut T,
    PhantomData<&'a mut U>,
    &'a mut U,
);

impl<'a, T, U> Reborrow for InterleavedMut<'a, T, U> {}

struct InterleavedRef<'a, T, U>(&'a T, PhantomData<&'a T>, &'a U, PhantomData<&'a U>);

impl<'a, T, U> Clone for InterleavedRef<'a, T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, U> Copy for InterleavedRef<'a, T, U> {}

impl<'a, T, U> CoerceShared<InterleavedRef<'a, T, U>> for InterleavedMut<'a, T, U> {}

fn read_source_leading<'a>(value: SourceLeadingRef<'a, i32>) -> &'a i32 {
    value.0
}

fn read_target_leading<'a>(value: TargetLeadingRef<'a, i32>) -> &'a i32 {
    value.1
}

fn read_interleaved<'a>(value: InterleavedRef<'a, i32, i64>) -> (&'a i32, &'a i64) {
    (value.0, value.2)
}

fn main() {
    let mut source_leading = 10;
    let wrapped = SourceLeadingMut(PhantomData, &mut source_leading);
    assert_eq!(*read_source_leading(wrapped), 10);

    let mut target_leading = 20;
    let wrapped = TargetLeadingMut(&mut target_leading);
    assert_eq!(*read_target_leading(wrapped), 20);

    let mut first = 30;
    let mut second = 40_i64;
    let wrapped = InterleavedMut(PhantomData, &mut first, PhantomData, &mut second);
    let (first, second) = read_interleaved(wrapped);
    assert_eq!((*first, *second), (30, 40));
}
