//@ known-bug: unknown
#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

// This test combines alias/projection normalization with the leaf `&mut T` to `&T`
// shared reborrow path.

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

type DirectInnerRef<'a, T> = InnerRef<'a, T>;

trait RefFamily<'a, T> {
    type Ref;
}

struct Projected;

impl<'a, T: 'a> RefFamily<'a, T> for Projected {
    type Ref = InnerRef<'a, T>;
}

type ProjectedInnerRef<'a, T> = <Projected as RefFamily<'a, T>>::Ref;

struct OuterMut<'a, T> {
    inner: InnerMut<'a, T>,
    tag: usize,
}

impl<'a, T> Reborrow for OuterMut<'a, T> {}

struct OuterAliasRef<'a, T> {
    inner: DirectInnerRef<'a, T>,
    tag: usize,
}

impl<'a, T> Clone for OuterAliasRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for OuterAliasRef<'a, T> {}

impl<'a, T> CoerceShared<OuterAliasRef<'a, T>> for OuterMut<'a, T> {}

struct OuterProjectionRef<'a, T: 'a> {
    inner: ProjectedInnerRef<'a, T>,
    tag: usize,
}

impl<'a, T: 'a> Clone for OuterProjectionRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: 'a> Copy for OuterProjectionRef<'a, T> {}

impl<'a, T: 'a> CoerceShared<OuterProjectionRef<'a, T>> for OuterMut<'a, T> {}

fn read_alias<'a>(outer: OuterAliasRef<'a, u32>) -> (&'a u32, usize) {
    (outer.inner.value, outer.tag)
}

fn read_projection<'a>(outer: OuterProjectionRef<'a, u32>) -> (&'a u32, usize) {
    (outer.inner.value, outer.tag)
}

const fn const_accept_projection(_outer: OuterProjectionRef<'_, u32>) {}

const fn consteval_projection_reborrow() {
    let mut value = 11;
    const_accept_projection(OuterMut {
        inner: InnerMut { value: &mut value },
        tag: 5,
    });
}

fn main() {
    const { consteval_projection_reborrow(); }

    let mut value = 22;
    let outer = OuterMut { inner: InnerMut { value: &mut value }, tag: 7 };

    let (alias_value, alias_tag) = read_alias(outer);
    assert_eq!((*alias_value, alias_tag), (22, 7));

    let (projection_value, projection_tag) = read_projection(outer);
    assert_eq!((*projection_value, projection_tag), (22, 7));
}
