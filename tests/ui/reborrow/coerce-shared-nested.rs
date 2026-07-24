// Invalid nested multi-field `CoerceShared` impls must be rejected without an ICE.

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct InnerMut<'a, T> {
    value: &'a mut T,
}

impl<'a, T> Reborrow for InnerMut<'a, T> {}

struct InnerRef<'a, T> {
    value: &'a T,
}

impl<'a, T> Clone for InnerRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for InnerRef<'a, T> {}

impl<'a, T> CoerceShared<InnerRef<'a, T>> for InnerMut<'a, T> {}

struct OuterMut<'a, T> {
    inner: InnerMut<'a, T>,
    tag: usize,
}

impl<'a, T> Reborrow for OuterMut<'a, T> {}

struct OuterRef<'a, T> {
    inner: InnerRef<'a, T>,
    tag: usize,
}

impl<'a, T> Clone for OuterRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for OuterRef<'a, T> {}

impl<'a, T> CoerceShared<OuterRef<'a, T>> for OuterMut<'a, T> {}
//~^ ERROR implementing `CoerceShared` does not allow multiple lifetimes or fields to be coerced

fn get<'a>(outer: OuterRef<'a, i32>) -> (&'a i32, usize) {
    (outer.inner.value, outer.tag)
}

fn main() {
    let mut value = 22;
    let outer = OuterMut { inner: InnerMut { value: &mut value }, tag: 7 };

    let (first, tag) = get(outer);
    assert_eq!((*first, tag), (22, 7));

    let (second, tag) = get(outer);
    assert_eq!((*second, tag), (22, 7));
}
