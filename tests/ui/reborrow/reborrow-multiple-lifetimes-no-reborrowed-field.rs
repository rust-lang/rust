//! Regression test for <https://github.com/rust-lang/rust/issues/165653>.
//!
//! A `Reborrow` type with no `&mut T` field and no `Reborrow` field without a `Copy`
//! impl is borrowed as a whole under its single lifetime, so it must not have more
//! than one.

#![feature(reborrow, trivial_bounds)]
#![allow(trivial_bounds)]

use std::marker::{PhantomData, Reborrow};

struct SharedOnly<'a, 'b> {
    a: &'a &'b str,
}

impl<'a, 'b> Reborrow for SharedOnly<'a, 'b> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

// The body from the issue. It never reaches borrowck, because coherence rejects the impl first.
impl<'a, 'b> SharedOnly<'a, 'b> {
    fn clone(&self) -> Self {
        *self
    }
}

struct PhantomOnly<'a, 'b>(PhantomData<(&'a (), &'b ())>);

impl<'a, 'b> Reborrow for PhantomOnly<'a, 'b> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

// Borrowck does not reborrow type parameter fields, even if they implement `Reborrow`.
struct ParamOnly<'a, 'b, T: Reborrow> {
    t: T,
    marker: PhantomData<(&'a (), &'b ())>,
}

impl<'a, 'b, T: Reborrow> Reborrow for ParamOnly<'a, 'b, T> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

trait HasAssoc<'a> {
    type Assoc;
}

impl<'a> HasAssoc<'a> for () {
    type Assoc = &'a mut ();
}

// Borrowck looks at the declared field type, so this field is not reborrowed even
// though it normalizes to `&mut ()`.
struct AssocOnly<'a, 'b> {
    assoc: <() as HasAssoc<'a>>::Assoc,
    shared: &'b (),
}

impl<'a, 'b> Reborrow for AssocOnly<'a, 'b> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

// `Wrap<'a, T>` is `Copy` when `T: Copy`, so a body with that bound copies `w` instead
// of reborrowing it. It must not count as a reborrowed field.
#[derive(Clone, Copy)]
struct Wrap<'a, T>(PhantomData<&'a ()>, T);

impl<'a, T: Reborrow> Reborrow for Wrap<'a, T> {}

struct MaybeCopyOnly<'a, 'b, T> {
    w: Wrap<'a, T>,
    shared: &'b (),
}

impl<'a, 'b, T: Reborrow> Reborrow for MaybeCopyOnly<'a, 'b, T> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

pub fn copy_bound<'a, 'b, T: Reborrow + Copy>(
    x: MaybeCopyOnly<'a, 'b, T>,
) -> MaybeCopyOnly<'a, 'b, T> {
    x
}

// The same type used with a concrete `Copy` + `Reborrow` type for `T`.
#[derive(Clone, Copy)]
struct CopyMarker<'a>(PhantomData<&'a ()>);

impl<'a> Reborrow for CopyMarker<'a> {}

pub fn concrete_copy<'a, 'b, 'c>(
    x: MaybeCopyOnly<'a, 'b, CopyMarker<'c>>,
) -> MaybeCopyOnly<'a, 'b, CopyMarker<'c>> {
    x
}

// `Pair<'a, 'b, T>` is only `Copy` because of the impl's `T: Copy` bound.
#[derive(Clone, Copy)]
struct Pair<'x, 'y, T>(PhantomData<(&'x (), &'y ())>, T);

struct CopyUnderImplBound<'a, 'b, T> {
    pair: Pair<'a, 'b, T>,
}

impl<'a, 'b, T: Copy> Reborrow for CopyUnderImplBound<'a, 'b, T> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

// `NotReborrow` has no `Copy` impl, but it is only accepted through the impl's
// where clause making it `Copy`, so it is copied and does not count.
struct NotReborrow<'x, 'y>(&'x (), &'y ());

struct CopyByWhereClause<'a, 'b> {
    inner: NotReborrow<'a, 'b>,
}

impl<'a, 'b> Reborrow for CopyByWhereClause<'a, 'b> where NotReborrow<'a, 'b>: Copy {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

fn main() {}
